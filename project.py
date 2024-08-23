import numpy as np
from mmhuman3d.core.visualization.visualize_smpl import _prepare_input_pose, _prepare_body_model, _prepare_mesh
from mmhuman3d.utils.demo_utils import convert_bbox_to_intrinsic
from typing import Iterable, List, Optional, Tuple, Union
import torch
from mmhuman3d.utils.demo_utils import convert_crop_cam_to_orig_img

from mmhuman3d.core.cameras.cameras import WeakPerspectiveCameras
from mmhuman3d.core.conventions.cameras.convert_convention import convert_K_4x4_to_3x3

idx = 100

bboxes = np.load('./result/inference_result.npz', allow_pickle=True)['bboxes_xyxy']
Ks = convert_bbox_to_intrinsic(bboxes, bbox_format='xyxy')
smpl_dict = np.load('./result/inference_result.npz', allow_pickle=True)['smpl'].reshape(-1)[0]
K = np.eye(3)
K[0][0] = 5000
K[1][1] = 5000
K[0][2] = 112
K[1][2] = 112
K = K@np.linalg.inv(K)@Ks[idx]@K
print(K)
R = np.eye(3)
global_orient = smpl_dict['global_orient'].reshape(-1, 3)
poses = smpl_dict['body_pose'].reshape(-1, 23*3)
betas = smpl_dict['betas'].reshape(-1, 10)
verts, poses, betas, transl = _prepare_input_pose(None, np.concatenate([global_orient, poses], axis=-1), betas, None)

body_model = _prepare_body_model(None, {'model_path': 'data/body_models/', 'type': 'smpl'})
vertices, joints, num_frames, num_person = _prepare_mesh(poses, betas, transl, verts, 0, None, body_model)

vertices = vertices[idx].detach().cpu().numpy()
print(vertices.min(axis=0), vertices.max(axis=0))
pred_cam = np.load('./result/inference_result.npz', allow_pickle=True)['pred_cams'][idx]

t = np.array([pred_cam[1], pred_cam[2], 5000 / (112 * pred_cam[0] + 1e-9)])
vertices_cam = vertices + t.reshape(-1, 3)
vertices_cam = (K@vertices_cam.T).T
vertices_projected = (vertices_cam/vertices_cam[:, -1:])[:,:2]
print(vertices_projected)
import cv2
img = cv2.imread(f'../bike/all_rgb/{str(idx).zfill(5)}.jpg')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
for idx_, loc in enumerate(vertices_projected):
    x = int(loc[0])
    y = int(loc[1])
    cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
    # cv2.putText(img, str(idx_), (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
cv2.imwrite('./pare_projected.png', img[:,:,::-1])