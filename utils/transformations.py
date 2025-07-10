import math
import numpy as np
import torch

deg2rad = lambda deg: deg * np.pi / 180


def normalize_vectors_np(vecs, eps=1e-8):
    assert vecs.ndim == 2
    assert vecs.shape[-1] == 3
    return vecs / (np.sqrt(np.sum(np.square(vecs), axis=-1, keepdims=True)) + eps)


def normalize_vectors_torch(vecs, eps=1e-8):
    assert vecs.ndim == 2
    assert vecs.shape[-1] == 3
    return vecs / (torch.sqrt(torch.sum(torch.square(vecs), dim=-1, keepdim=True)) + eps)


def batch_rodrigues(theta):
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat_to_rotmat(quat)


def quat_to_rotmat(quat):
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(
        B, 3, 3
    )
    return rotMat


def convert_Kparams_Kmat(tensor):
    K_elem = torch.split(tensor, 1)
    elem1 = torch.Tensor([1, 0, 0]).to(tensor.device) * K_elem[0]
    elem2 = torch.Tensor([0, 1, 0]).to(tensor.device) * K_elem[1]
    elem3 = torch.Tensor([1, 0, 0]).to(tensor.device) * K_elem[2] + torch.Tensor([0, 1, 0]).to(tensor.device) * K_elem[3] + torch.Tensor([0, 0, 1]).to(tensor.device)

    elem1 = elem1.unsqueeze(0).view(3, -1)
    elem2 = elem2.unsqueeze(0).view(3, -1)
    elem3 = elem3.unsqueeze(0).view(3, -1)

    result = torch.cat((elem1, elem2, elem3), dim=1)
    return result


def batch_convert_Kparams_Kmat(K_elem_batch):
    # K_elem_batch: Bx4
    assert K_elem_batch.ndim == 2
    assert K_elem_batch.shape[-1] == 4

    # elem1, elem2, elem3 --> Bx3
    elem1 = torch.Tensor([[1, 0, 0]]).to(K_elem_batch.device) * K_elem_batch[:, 0:1]
    elem2 = torch.Tensor([[0, 1, 0]]).to(K_elem_batch.device) * K_elem_batch[:, 1:2]
    elem3 = (
        torch.Tensor([[1, 0, 0]]).to(K_elem_batch.device) * K_elem_batch[:, 2:3]
        + torch.Tensor([[0, 1, 0]]).to(K_elem_batch.device) * K_elem_batch[:, 3:4]
        + torch.Tensor([[0, 0, 1]]).to(K_elem_batch.device)
    )

    elem1 = elem1.unsqueeze(1).permute(0, 2, 1)  # Bx3x1
    elem2 = elem2.unsqueeze(1).permute(0, 2, 1)  # Bx3x1
    elem3 = elem3.unsqueeze(1).permute(0, 2, 1)  # Bx3x1

    K = torch.cat((elem1, elem2, elem3), dim=-1)
    return K


def get_focal_length(img_length, fovy, use_radian=False):
    if not use_radian:
        fovy = fovy * math.pi / 180.0
    return img_length / (2.0 * math.tan(fovy / 2))


def rotmat_to_axis_angle(matrix):
    # Axes.
    axis = torch.zeros(3, dtype=torch.float32)
    axis[0] = matrix[2, 1] - matrix[1, 2]
    axis[1] = matrix[0, 2] - matrix[2, 0]
    axis[2] = matrix[1, 0] - matrix[0, 1]

    # Angle.
    r = torch.hypot(axis[0], torch.hypot(axis[1], axis[2]))
    t = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    theta = torch.atan2(r, t - 1)

    # Normalise the axis.
    axis = axis / r

    # Return the data.
    return axis, theta


def rotation_matrix_to_angle_axis(rotation_matrix):
    # todo add check that matrix is a valid rotation matrix
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError("Input size must be a three dimensional tensor. Got {}".format(rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError("Input size must be a N x 3 x 4  tensor. Got {}".format(rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1], t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] + rmat_t[:, 1, 0], t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2], rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1], rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2.float() * mask_d0_d1.float()
    mask_c1 = mask_d2.float() * (1 - mask_d0_d1.float())
    mask_c2 = (1 - mask_d2.float()) * mask_d0_nd1.float()
    mask_c3 = (1 - mask_d2.float()) * (1 - mask_d0_nd1.float())

    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 + t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa  # noqa
    q *= 0.5
    return q


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}".format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta), torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis