import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import os
import json
import glob
from typing import Dict, List, Optional

# --- Geometric Utility Functions ---

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Calculate Euclidean distance between each two points."""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Input: points [B, N, C], idx [B, S]. Return: indexed points [B, S, C]"""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    idx = torch.clamp(idx, 0, points.shape[1] - 1)
    return points[batch_indices, idx, :]

def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Iterative farthest point sampling (FPS)"""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """Group points within a local sphere"""
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return torch.clamp(group_idx, 0, N - 1)

# --- PointNet++ Modules ---

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None: points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = self._sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = self._sample_and_group(xyz, points)
        
        new_points = new_points.permute(0, 3, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        return new_xyz.permute(0, 2, 1), new_points

    def _sample_and_group(self, xyz, points):
        B, N, C = xyz.shape
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, self.npoint, 1, C)
        if points is not None:
            grouped_points = index_points(points, idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz_norm
        return new_xyz, new_points

    def _sample_and_group_all(self, xyz, points):
        device = xyz.device
        B, N, C = xyz.shape
        new_xyz = torch.zeros(B, 1, C).to(device)
        grouped_xyz = xyz.view(B, 1, N, C)
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1) if points is not None else grouped_xyz
        return new_xyz, new_points

# --- Main Model ---

class DINOv2RGBDPoseModel(nn.Module):
    def __init__(self, num_objects=15, dinov2_model='dinov2_vitb14', freeze_backbone=False, pointnet_dim=1024):
        super().__init__()
        
        # 1. RGB Backbone (DINOv2)
        self.dinov2_rgb = torch.hub.load('facebookresearch/dinov2', dinov2_model)
        if freeze_backbone:
            for param in self.dinov2_rgb.parameters(): param.requires_grad = False
        
        dims = {'dinov2_vits14': 384, 'dinov2_vitb14': 768, 'dinov2_vitl14': 1024, 'dinov2_vitg14': 1536}
        self.rgb_feature_dim = dims.get(dinov2_model, 768)

        # 2. Depth Backbone (PointNet++)
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, 3+3, [64, 64, 128], False)
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128+3, [128, 128, 256], False)
        self.sa3 = PointNetSetAbstraction(None, None, None, 256+3, [256, 512, pointnet_dim], True)

        # 3. Fusion & Heads
        self.fusion = nn.Sequential(
            nn.Linear(self.rgb_feature_dim + pointnet_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.rotation_head = nn.Linear(512, 9)
        self.translation_head = nn.Linear(512, 3)
        self.obj_head = nn.Linear(512, num_objects)

    def _get_pc_features(self, xyz):
        # xyz shape: [B, 3, N]
        l1_xyz, l1_pts = self.sa1(xyz, None)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        _, l3_pts = self.sa3(l2_xyz, l2_pts)
        return l3_pts.squeeze(-1)

    def forward(self, rgb, pc):
        """
        rgb: [B, 3, 224, 224]
        pc:  [B, 3, N] (Point cloud)
        """
        feat_rgb = self.dinov2_rgb(rgb)
        feat_depth = self._get_pc_features(pc)
        
        fused = torch.cat([feat_rgb, feat_depth], dim=1)
        fused = self.fusion(fused)
        
        rot = self.rotation_head(fused).view(-1, 3, 3)
        # SVD Orthogonalization
        u, _, vt = torch.svd(rot)
        rot = torch.bmm(u, vt)
        
        return {
            'rotation': rot,
            'translation': self.translation_head(fused),
            'logits': self.obj_head(fused)
        }

# --- Feature Extraction Test ---

def test_variants():
    print(f"{'Variant':<20} | {'RGB Dim':<10} | {'Output Status'}")
    print("-" * 50)
    variants = ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14']
    
    for v in variants:
        try:
            model = DINOv2RGBDPoseModel(dinov2_model=v, freeze_backbone=True).cuda()
            rgb = torch.randn(1, 3, 224, 224).cuda()
            pc = torch.randn(1, 3, 2048).cuda()
            
            with torch.no_grad():
                out = model(rgb, pc)
            
            print(f"{v:<20} | {model.rgb_feature_dim:<10} | SUCCESS (Rot: {out['rotation'].shape})")
        except Exception as e:
            print(f"{v:<20} | Error: {str(e)}")

if __name__ == "__main__":
    test_variants()
