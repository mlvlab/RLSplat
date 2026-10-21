import torch
from jaxtyping import Float
from torch import Tensor


def camera_normalization(pivotal_pose: torch.Tensor, poses: torch.Tensor):

    canonical_camera_extrinsics = torch.tensor([[
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]], dtype=torch.float32, device=pivotal_pose.device)
    pivotal_pose_inv = torch.inverse(pivotal_pose)
    camera_norm_matrix = torch.bmm(canonical_camera_extrinsics, pivotal_pose_inv)

    poses = torch.bmm(camera_norm_matrix.repeat(poses.shape[0], 1, 1), poses)

    return poses



def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def update_pose(cam_trans_delta: Float[Tensor, "batch 3"],
                cam_rot_delta: Float[Tensor, "batch 3"],
                extrinsics: Float[Tensor, "batch 4 4"],
                ):
    bs = cam_trans_delta.shape[0]

    tau = torch.cat([cam_trans_delta, cam_rot_delta], dim=-1)
    T_w2c = extrinsics.inverse()

    new_w2c_list = []
    for i in range(bs):
        new_w2c = SE3_exp(tau[i]) @ T_w2c[i]
        new_w2c_list.append(new_w2c)

    new_w2c = torch.stack(new_w2c_list, dim=0)
    return new_w2c.inverse()
