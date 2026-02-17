class Compact3DGaussian(nn.Module):

    def __init__(self, in_channels, num_gaussians=32, feature_dim=128):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.feature_dim = feature_dim
        
        # Predictor for: Means(3), Covariance(6), Opacity/Weight(1), Feature(dim)
        self.param_dim = 3 + 6 + 1 + feature_dim 
        
        self.predictor = nn.Sequential(
            nn.Linear(in_channels, 256), nn.ReLU(),
            nn.Linear(256, num_gaussians * self.param_dim)
        )
        
    def forward(self, xyz, features):
        B, N, _ = xyz.shape
        # Global context for Gaussian prediction
        global_context = torch.max(features, dim=1)[0]
        params = self.predictor(global_context).view(B, self.num_gaussians, -1)
        
        means = params[:, :, :3]
        weights = torch.sigmoid(params[:, :, 9:10])
        g_features = params[:, :, 10:]
        
        # Simplified radial basis influence for feature aggregation
        # Distances between points and Gaussian centers
        dists = torch.cdist(xyz, means) # [B, N, num_gaussians]
        influence = torch.exp(-dists) * weights.transpose(1, 2)
        
        # Aggregate features into Gaussian tokens
        # [B, num_gaussians, N] @ [B, N, C] -> [B, num_gaussians, C]
        aggregated_features = torch.bmm(influence.transpose(1, 2), features)
        
        return aggregated_features, means

class C3GPoseModel(nn.Module):
    def __init__(self, num_objects=15, dinov2_variant='dinov2_vitb14'):
        super().__init__()
        
        # RGB Stream
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', dinov2_variant)
        rgb_dim = 768 # for ViT-B
        
        # Depth Stream (PTv3 + C3G)
        self.c3g = Compact3DGaussian(in_channels=1024, num_gaussians=64, feature_dim=512)
        
        # Object Query Embedding
        self.obj_query = nn.Embedding(num_objects + 1, 512)
        
        # Fusion and Heads
        self.cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.rgb_proj = nn.Linear(rgb_dim, 512)
        
        self.rot_head = nn.Linear(512, 9)
        self.trans_head = nn.Linear(512, 3)
        self.uncertainty_head = nn.Linear(512, 1) # ALEATORIC UNCERTAINTY

    def forward(self, rgb, pc_xyz, pc_features, obj_id):
        B = rgb.shape[0]
        
        # 1. RGB Features
        feat_rgb = self.rgb_proj(self.dinov2(rgb)) # [B, 512]
        
        # 2. C3G Geometric Tokens
        feat_geo, _ = self.c3g(pc_xyz, pc_features) # [B, 64, 512]
        
        # 3. Object-Centric Alignment
        query = self.obj_query(obj_id).unsqueeze(1) # [B, 1, 512]
        # Query attends to geometric tokens
        aligned_feat, _ = self.cross_attn(query, feat_geo, feat_geo)
        
        # 4. Final Prediction
        fused = aligned_feat.squeeze(1) + feat_rgb
        
        # Orthogonalize Rotation
        raw_rot = self.rot_head(fused).view(B, 3, 3)
        U, _, V = torch.linalg.svd(raw_rot)
        rotation = torch.bmm(U, V)
        
        return {
            'rotation': rotation,
            'translation': self.trans_head(fused),
            'uncertainty': torch.exp(self.uncertainty_head(fused))
        }
