class PoseVisualizer3D:
    """Handles 3D to 2D projections and drawing of bounding boxes and axes."""
    
    def __init__(self):
        # LM-O dataset object dimensions (mm)
        self.obj_dims = {
            1:  {'name': 'ape',         'size': [102, 78, 138]},
            2:  {'name': 'benchvise',   'size': [247, 135, 187]},
            5:  {'name': 'can',         'size': [100, 120, 100]},
            6:  {'name': 'cat',         'size': [125, 85, 175]},
            8:  {'name': 'driller',     'size': [245, 175, 85]},
            9:  {'name': 'duck',        'size': [109, 95, 125]},
            10: {'name': 'eggbox',      'size': [165, 115, 75]},
            11: {'name': 'glue',        'size': [109, 175, 60]},
            12: {'name': 'holepuncher', 'size': [146, 73, 200]}
        }

    def get_bbox_corners(self, obj_id: int) -> np.ndarray:
        """Returns 8 corners of the 3D bbox in object coordinates."""
        size = self.obj_dims.get(obj_id, {'size': [100, 100, 100]})['size']
        w, h, d = [s/2 for s in size]
        return np.array([
            [-w,-h,-d], [w,-h,-d], [w,h,-d], [-w,h,-d],
            [-w,-h,d],  [w,-h,d],  [w,h,d],  [-w,h,d]
        ])

    def project(self, pts_3d: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Projects 3D points to 2D pixels using Camera Matrix K."""
        pts_cam = (R @ pts_3d.T) + t.reshape(3, 1)
        # Perspective division
        pts_2d = K @ pts_cam
        pts_2d = pts_2d[:2, :] / pts_2d[2, :]
        return pts_2d.T.astype(int)

    def draw_bbox(self, img: np.ndarray, pts_2d: np.ndarray, color=(0, 255, 0), thick=2):
        """Draws a wireframe 3D bounding box."""
        edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
        for i, j in edges:
            cv2.line(img, tuple(pts_2d[i]), tuple(pts_2d[j]), color, thick)
        # Highlight front face
        for i, j in [(4,5), (5,6), (6,7), (7,4)]:
            cv2.line(img, tuple(pts_2d[i]), tuple(pts_2d[j]), (0, 0, 255), thick + 1)
        return img

    def draw_axes(self, img: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray, scale=50):
        """Draws RGB coordinate axes at the object origin."""
        pts_3d = np.array([[0,0,0], [scale,0,0], [0,scale,0], [0,0,scale]])
        pts_2d = self.project(pts_3d, R, t, K)
        origin = tuple(pts_2d[0])
        cv2.arrowedLine(img, origin, tuple(pts_2d[1]), (0,0,255), 2) # X - Red
        cv2.arrowedLine(img, origin, tuple(pts_2d[2]), (0,255,0), 2) # Y - Green
        cv2.arrowedLine(img, origin, tuple(pts_2d[3]), (255,0,0), 2) # Z - Blue
        return img

# --- 2. Data & Evaluation Utilities ---

class LMOEvaluator:
    """Handles dataset loading and statistical error calculation."""
    
    def __init__(self, root: str):
        self.root = root
        self.vis = PoseVisualizer3D()

    def get_samples(self) -> List[Dict]:
        """Parses BOP format folders for RGB, GT, and Camera info."""
        samples = []
        scenes = [d for d in os.listdir(self.root) if d.isdigit() or d.startswith('00')]
        
        for scene in scenes:
            scene_path = os.path.join(self.root, scene)
            with open(os.path.join(scene_path, 'scene_gt.json')) as f: gts = json.load(f)
            with open(os.path.join(scene_path, 'scene_camera.json')) as f: cams = json.load(f)
            
            for img_id, obj_list in gts.items():
                img_path = os.path.join(scene_path, 'rgb', f"{int(img_id):06d}.png")
                K = np.array(cams[img_id]['cam_K']).reshape(3, 3)
                samples.append({'path': img_path, 'objs': obj_list, 'K': K, 'id': img_id})
        return samples

    @staticmethod
    def calculate_errors(pred: Dict, gt: Dict) -> Tuple[float, float]:
        """Returns Translation Error (mm) and Rotation Error (Frobenius)."""
        t_err = np.linalg.norm(pred['t'] - gt['t'])
        r_err = np.linalg.norm(pred['R'] - gt['R'])
        return t_err, r_err

# --- 3. Main Execution Workflow ---

def run_batch_visualization(data_root: str, out_dir: str, limit=10):
    os.makedirs(out_dir, exist_ok=True)
    evaluator = LMOEvaluator(data_root)
    samples = evaluator.get_samples()
    stats = {'t_err': [], 'r_err': [], 'ids': []}

    print(f"Processing {min(limit, len(samples))} samples...")

    for i, sample in enumerate(samples[:limit]):
        img = cv2.imread(sample['path'])
        K = sample['K']

        for obj in sample['objs']:
            obj_id = obj['obj_id']
            gt_pose = {'R': np.array(obj['cam_R_m2c']).reshape(3,3), 't': np.array(obj['cam_t_m2c'])}
            
            # Simulated Prediction (Replace with actual model.predict() here)
            pred_pose = {
                'R': gt_pose['R'] + np.random.normal(0, 0.05, (3,3)), 
                't': gt_pose['t'] + np.random.normal(0, 30, 3)
            }
            # Re-orthogonalize R
            u, _, vt = np.linalg.svd(pred_pose['R']); pred_pose['R'] = u @ vt

            # Metrics
            te, re = evaluator.calculate_errors(pred_pose, gt_pose)
            stats['t_err'].append(te); stats['r_err'].append(re); stats['ids'].append(obj_id)

            # Draw
            corners = evaluator.vis.get_bbox_corners(obj_id)
            gt_2d = evaluator.vis.project(corners, gt_pose['R'], gt_pose['t'], K)
            pred_2d = evaluator.vis.project(corners, pred_pose['R'], pred_pose['t'], K)
            
            img = evaluator.vis.draw_bbox(img, gt_2d, (0, 255, 0), 2)    # Green GT
            img = evaluator.vis.draw_bbox(img, pred_2d, (0, 0, 255), 1)  # Red Pred
            img = evaluator.vis.draw_axes(img, gt_pose['R'], gt_pose['t'], K)

        cv2.imwrite(os.path.join(out_dir, f"res_{i:03d}.png"), img)

    # Plot Results
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.hist(stats['t_err'], bins=20, color='skyblue'); plt.title("Trans Error (mm)")
    plt.subplot(1, 2, 2); plt.hist(stats['r_err'], bins=20, color='salmon'); plt.title("Rot Error (Frob)")
    plt.savefig(os.path.join(out_dir, 'stats.png'))
    print(f"Done! Results saved to {out_dir}")

if __name__ == "__main__":
    # Update this path to your local dataset
    DATA_PATH = "D:/3D_Human_Pose/lmo_test_bop19"
    run_batch_visualization(DATA_PATH, "pose_results")
