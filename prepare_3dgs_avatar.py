import numpy as np
from mmhuman3d.utils.demo_utils import convert_bbox_to_intrinsic
import torch
from human_body_prior.body_model.body_model import BodyModel
import os, json, shutil

def get_center_and_diag(cam_centers):
    cam_centers = np.hstack(cam_centers)
    avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    return center.flatten(), diagonal

bboxes = np.load('./result/inference_result.npz', allow_pickle=True)['bboxes_xyxy']
Ks = convert_bbox_to_intrinsic(bboxes, bbox_format='xyxy')
out_dir = '../neuman_preprocessed/bike'
num_views = len(Ks)
train_views = int(num_views*0.9)

smpl_out_dir = os.path.join(out_dir, 'models')
if not os.path.exists(smpl_out_dir):
    os.makedirs(smpl_out_dir)

if not os.path.exists(os.path.join(out_dir, 'train')):
    os.makedirs(os.path.join(out_dir, 'train'))
if not os.path.exists(os.path.join(out_dir, 'test')):
    os.makedirs(os.path.join(out_dir, 'test'))

smpl_dict = np.load('./result/inference_result.npz', allow_pickle=True)['smpl'].reshape(-1)[0]

global_orient = smpl_dict['global_orient'].reshape(-1, 3)
poses = smpl_dict['body_pose'].reshape(-1, 23*3)
betas = smpl_dict['betas'].reshape(-1, 10)
# import pdb
# pdb.set_trace()
# verts, poses, betas, transl = _prepare_input_pose(None, np.concatenate([global_orient, poses], axis=-1), betas, None)

# body_model = _prepare_body_model(None, {'model_path': 'data/body_models/', 'type': 'smpl'})
# vertices, joints, num_frames, num_person = _prepare_mesh(poses, betas, transl, verts, 0, None, body_model)
# vertices_min = vertices.min(axis=1)
# vertices_max = vertices.max(axis=1)
# print(vertices.shape)
# exit()

body_model = BodyModel(bm_path='./data/body_models/smpl/SMPL_NEUTRAL.pkl', num_betas=10, batch_size=1).cuda()

faces = np.load('../3dgs-avatar-release/body_models/misc/faces.npz')['faces']
cam_names = list(range(0, num_views))
cam_names = [str(cam_name) for cam_name in cam_names]
all_cam_params = {'all_cam_names': cam_names}
cam_centers = []
for cam_idx, cam_name in enumerate(cam_names):
    K = np.eye(3)
    K[0][0] = 5000
    K[1][1] = 5000
    K[0][2] = 112
    K[1][2] = 112
    K = Ks[cam_idx]@K
    # K = K@np.linalg.inv(K)@Ks[cam_idx]@K
    # print(K)
    R = np.eye(3)
    D = np.zeros((5,1))
    
    # vertices_frame = vertices[cam_idx].detach().cpu().numpy()
    pred_cam = np.load('./result/inference_result.npz', allow_pickle=True)['pred_cams'][cam_idx]

    T = np.array([pred_cam[1], pred_cam[2], 5000 / (112 * pred_cam[0] + 1e-9)])
    cam_center = -R.T@T.reshape(3,1)
    cam_centers.append(cam_center)
    
    body = body_model(betas=torch.tensor(betas[cam_idx].reshape(1,-1)).cuda())
    minimal_shape = body.v.detach().cpu().numpy()[0]
    
    body = body_model(root_orient=torch.tensor(global_orient[cam_idx].reshape(1,3)).cuda(), pose_body=torch.tensor(poses[cam_idx][:63].reshape(1,-1)).cuda(), pose_hand=torch.tensor(poses[cam_idx][63:]).reshape(1,-1).cuda(), betas=torch.tensor(betas[cam_idx].reshape(1,-1)).cuda(), trans=torch.zeros((1,3)).cuda())
    
    # import pdb
    # pdb.set_trace()
    
    
    # vertices_cam = body.v.reshape(-1,3).detach().cpu().numpy() + T.reshape(-1, 3)
    # vertices_cam = (K@vertices_cam.T).T
    # vertices_projected = (vertices_cam/vertices_cam[:, -1:])[:,:2]
    # # print(vertices_projected)
    # import cv2
    # img = cv2.imread(f'../bike/all_rgb/{str(cam_idx).zfill(5)}.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # for idx_, loc in enumerate(vertices_projected):
    #     x = int(loc[0])
    #     y = int(loc[1])
    #     cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
        
    # cv2.imwrite('./pare_projected.png', img[:,:,::-1])
    # exit()
    
    bone_transforms = body.bone_transforms.detach().cpu().numpy()
    Jtr_posed = body.Jtr.detach().cpu().numpy()
    out_filename = os.path.join(smpl_out_dir, '{:06d}.npz'.format(cam_idx))
    np.savez(out_filename,
             minimal_shape=minimal_shape,
             betas=betas[cam_idx].reshape(1,-1)[0],
             Jtr_posed=Jtr_posed[0],
             bone_transforms=bone_transforms[0],
             trans=np.zeros(3,),
             root_orient=global_orient[cam_idx].reshape(1,3)[0],
             pose_body=poses[cam_idx][:63],
             pose_hand=poses[cam_idx][63:])
    
    cam_params = {'K': K.tolist(), 'D': D.tolist(), 'R': R.tolist(), 'T': T.reshape(3,1).tolist()}
    all_cam_params.update({cam_name: cam_params})
    if cam_idx < train_views:
        shutil.copy(f'../bike/all_rgb/{str(cam_idx).zfill(5)}.jpg', os.path.join(os.path.join(out_dir, 'train'), '{:06d}.jpg'.format(cam_idx)))
        shutil.copy(f'../bike/all_rgb/{str(cam_idx).zfill(5)}.png', os.path.join(os.path.join(out_dir, 'train'), '{:06d}.png'.format(cam_idx)))
    else:
        shutil.copy(f'../bike/all_rgb/{str(cam_idx).zfill(5)}.jpg', os.path.join(os.path.join(out_dir, 'test'), '{:06d}.jpg'.format(cam_idx)))
        shutil.copy(f'../bike/all_rgb/{str(cam_idx).zfill(5)}.png', os.path.join(os.path.join(out_dir, 'test'), '{:06d}.png'.format(cam_idx)))

cam_centers = np.array(cam_centers)
center, diagonal = get_center_and_diag(cam_centers)
radius = diagonal * 1.1
print(radius)
# exit()
with open(os.path.join(out_dir, 'cam_params.json'), 'w') as f:
        json.dump(all_cam_params, f)