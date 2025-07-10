import sys
import os
import torch
import numpy as np

from scipy import integrate
from ipdb import set_trace
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from imports.genpose.utils.genpose_utils import get_pose_dim, get_pose_dim_RT_sequence
from imports.genpose.utils.misc import normalize_rotation, get_rot_matrix

def global_prior_likelihood(z, sigma_max):
    """The likelihood of a Gaussian distribution with mean zero and 
        standard deviation sigma."""
    # z: [bs, pose_dim]
    shape = z.shape
    N = np.prod(shape[1:]) # pose_dim
    return -N / 2. * torch.log(2*np.pi*sigma_max**2) - torch.sum(z**2, dim=-1) / (2 * sigma_max**2)


def cond_ode_likelihood(
        score_model,
        data,
        prior,
        sde_coeff,
        marginal_prob_fn,
        atol=1e-5, 
        rtol=1e-5, 
        device='cuda', 
        eps=1e-5,
        num_steps=None,
        pose_mode='quat_wxyz', 
        init_x=None,
    ):
    
    pose_dim = get_pose_dim_RT_sequence(pose_mode)
    batch_size = data['pts'].shape[0]
    epsilon = prior((batch_size, pose_dim)).to(device)
    init_x = data['sampled_pose'].clone().cpu().numpy() if init_x is None else init_x
    shape = init_x.shape
    init_logp = np.zeros((shape[0],)) # [bs]
    init_inp = np.concatenate([init_x.reshape(-1), init_logp], axis=0)
    
    def score_eval_wrapper(data):
        """A wrapper of the score-based model for use by the ODE solver."""
        with torch.no_grad():
            score = score_model(data)
        return score.cpu().numpy().reshape((-1,))

    def divergence_eval(data, epsilon):      
        """Compute the divergence of the score-based model with Skilling-Hutchinson."""
        # save ckpt of sampled_pose
        origin_sampled_pose = data['sampled_pose'].clone()
        with torch.enable_grad():
            # make sampled_pose differentiable
            data['sampled_pose'].requires_grad_(True)
            score = score_model(data)
            score_energy = torch.sum(score * epsilon) # [, ]
            grad_score_energy = torch.autograd.grad(score_energy, data['sampled_pose'])[0] # [bs, pose_dim]
        # reset sampled_pose
        data['sampled_pose'] = origin_sampled_pose
        return torch.sum(grad_score_energy * epsilon, dim=-1) # [bs, 1]
    
    def divergence_eval_wrapper(data):
        """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
        with torch.no_grad(): 
            # Compute likelihood.
            div = divergence_eval(data, epsilon) # [bs, 1]
        return div.cpu().numpy().reshape((-1,)).astype(np.float64)
    
    def ode_func(t, inp):        
        """The ODE function for use by the ODE solver."""
        # split x, logp from inp
        x = inp[:-shape[0]]
        logp = inp[-shape[0]:] # haha, actually we do not need use logp here
        # calc x-grad
        x = torch.tensor(x.reshape(-1, pose_dim), dtype=torch.float32, device=device)
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        drift, diffusion = sde_coeff(torch.tensor(t))
        drift = drift.cpu().numpy()
        diffusion = diffusion.cpu().numpy()
        data['sampled_pose'] = x
        data['t'] = time_steps
        x_grad = drift - 0.5 * (diffusion**2) * score_eval_wrapper(data)
        # calc logp-grad
        logp_grad = drift - 0.5 * (diffusion**2) * divergence_eval_wrapper(data)
        # concat curr grad
        return  np.concatenate([x_grad, logp_grad], axis=0)
  
    # Run the black-box ODE solver, note the 
    res = integrate.solve_ivp(ode_func, (eps, 1.0), init_inp, rtol=rtol, atol=atol, method='RK45')
    zp = torch.tensor(res.y[:, -1], device=device) # [bs * (pose_dim + 1)]
    z = zp[:-shape[0]].reshape(shape) # [bs, pose_dim]
    delta_logp = zp[-shape[0]:].reshape(shape[0]) # [bs,] logp
    _, sigma_max = marginal_prob_fn(None, torch.tensor(1.).to(device)) # we assume T = 1 
    prior_logp = global_prior_likelihood(z, sigma_max)
    log_likelihoods = (prior_logp + delta_logp) / np.log(2) # negative log-likelihoods (nlls)
    return z, log_likelihoods


def cond_pc_sampler(
        score_model, 
        data,
        prior,
        sde_coeff,
        num_steps=500, 
        snr=0.16,                
        device='cuda',
        eps=1e-5,
        pose_mode='quat_wxyz',
        init_x=None,
    ):
    
    pose_dim = get_pose_dim_RT_sequence(pose_mode)
    batch_size = data['pts'].shape[0]
    init_x = prior((batch_size, pose_dim)).to(device) if init_x is None else init_x
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    noise_norm = np.sqrt(pose_dim) 
    x = init_x
    poses = []
    with torch.no_grad():
        for time_step in time_steps:      
            batch_time_step = torch.ones(batch_size, device=device).unsqueeze(-1) * time_step
            # Corrector step (Langevin MCMC)
            data['sampled_pose'] = x
            data['t'] = batch_time_step
            grad = score_model(data)
            grad_norm = torch.norm(grad.reshape(batch_size, -1), dim=-1).mean()
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)  

            # normalisation
            if pose_mode == 'quat_wxyz' or pose_mode == 'quat_xyzw':
                # quat, should be normalised
                x[:, :4] /= torch.norm(x[:, :4], dim=-1, keepdim=True)   
            elif pose_mode == 'euler_xyz':
                pass
            else:
                # rotation(x axis, y axis), should be normalised
                x[:, :3] /= torch.norm(x[:, :3], dim=-1, keepdim=True)
                x[:, 3:6] /= torch.norm(x[:, 3:6], dim=-1, keepdim=True)

            # Predictor step (Euler-Maruyama)
            drift, diffusion = sde_coeff(batch_time_step)
            drift = drift - diffusion**2*grad # R-SDE
            mean_x = x + drift * step_size
            x = mean_x + diffusion * torch.sqrt(step_size) * torch.randn_like(x)
            
            # normalisation
            x[:, :-3] = normalize_rotation(x[:, :-3], pose_mode)
            poses.append(x.unsqueeze(0))
    
    xs = torch.cat(poses, dim=0)
    xs[:, :, -3:] += data['pts_center'].unsqueeze(0).repeat(xs.shape[0], 1, 1)
    mean_x[:, -3:] += data['pts_center']
    mean_x[:, :-3] = normalize_rotation(mean_x[:, :-3], pose_mode)
    # The last step does not include any noise
    return xs.permute(1, 0, 2), mean_x 


def cond_ode_sampler(
        score_model,
        data,
        prior,
        sde_coeff,
        atol=1e-5, 
        rtol=1e-5, 
        device='cuda', 
        eps=1e-5,
        T=1.0,
        num_steps=None,
        pose_mode='quat_wxyz', 
        denoise=True,
        init_x=None,
    ):
    pose_dim = get_pose_dim_RT_sequence(pose_mode)
    batch_size=data['pts'].shape[0]
    init_x = prior((batch_size, pose_dim), T=T).to(device) if init_x is None else init_x + prior((batch_size, pose_dim), T=T).to(device)
    shape = init_x.shape
    
    def score_eval_wrapper(data):
        """A wrapper of the score-based model for use by the ODE solver."""
        with torch.no_grad():
            score = score_model(data)
        return score.cpu().numpy().reshape((-1,))
    
    def ode_func(t, x):      
        """The ODE function for use by the ODE solver."""
        x = torch.tensor(x.reshape(-1, pose_dim), dtype=torch.float32, device=device)
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        drift, diffusion = sde_coeff(torch.tensor(t))
        drift = drift.cpu().numpy()
        diffusion = diffusion.cpu().numpy()
        data['sampled_pose'] = x
        data['t'] = time_steps
        return drift - 0.5 * (diffusion**2) * score_eval_wrapper(data)
  
    # Run the black-box ODE solver, note the 
    t_eval = None
    if num_steps is not None:
        # num_steps, from T -> eps
        t_eval = np.linspace(T, eps, num_steps)
    res = integrate.solve_ivp(ode_func, (T, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45', t_eval=t_eval)
    xs = torch.tensor(res.y, device=device).T.view(-1, batch_size, pose_dim) # [num_steps, bs, pose_dim]
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape) # [bs, pose_dim]
    # denoise, using the predictor step in P-C sampler
    if denoise:
        # Reverse diffusion predictor for denoising
        vec_eps = torch.ones((x.shape[0], 1), device=x.device) * eps
        drift, diffusion = sde_coeff(vec_eps)
        data['sampled_pose'] = x.float()
        data['t'] = vec_eps
        grad = score_model(data)
        drift = drift - diffusion**2*grad       # R-SDE
        mean_x = x + drift * ((1-eps)/(1000 if num_steps is None else num_steps))
        x = mean_x
    
    num_steps = xs.shape[0]
    xs = xs.reshape(batch_size*num_steps, -1)
    xs[:, :-3] = normalize_rotation(xs[:, :-3], pose_mode)
    xs = xs.reshape(num_steps, batch_size, -1)
    xs[:, :, -3:] += data['pts_center'].unsqueeze(0).repeat(xs.shape[0], 1, 1)
    x[:, :-3] = normalize_rotation(x[:, :-3], pose_mode)
    x[:, -3:] += data['pts_center']
    return xs.permute(1, 0, 2), x

# TRAINING
def cond_ode_sampler_for_RT(
        score_model,
        data,
        prior,
        sde_coeff,
        atol=1e-5, 
        rtol=1e-5, 
        device='cuda', 
        eps=1e-5,
        T=1.0,
        num_steps=None,
        pose_mode='quat_wxyz',
        denoise=True,
        init_x=None,
    ):
    pose_dim = get_pose_dim_RT_sequence(pose_mode)
    # batch_size=data['pts'].shape[0]
    batch_size=data['thetas'].shape[0]
    init_x = prior((batch_size, pose_dim), T=T).to(device) if init_x is None else init_x + prior((batch_size, pose_dim), T=T).to(device)
    shape = init_x.shape
    
    def smoothness_loss(x):
        position_smoothness_loss = torch.nn.functional.mse_loss(x[1:, :], x[:-1, :])
        velocity_smoothness_loss = torch.nn.functional.mse_loss((x[2:, :] - x[1:-1, :]), (x[1:-1, :] - x[:-2, :]))

        # return (position_smoothness_loss + velocity_smoothness_loss) * 10000
        return (position_smoothness_loss) * 10000
    
    def contact_loss(x):
        def find_close_points(A, B, threshold=0.1):
            A_expanded = A.unsqueeze(1)  # Shape: Nx1x3
            B_expanded = B.unsqueeze(0)  # Shape: 1xMx3

            # Compute the squared Euclidean distances between points in A and B
            distances = torch.norm(A_expanded - B_expanded, dim=2)  # Shape: NxM

            # Find indices where distance is less than threshold
            indices_A, indices_B = torch.where(distances < threshold)
            unique_indices_A = torch.unique(indices_A)
            unique_indices_B = torch.unique(indices_B)

            A_subset = A[unique_indices_A]
            B_subset = B[unique_indices_B]

            print(unique_indices_A.shape)
            print(unique_indices_B.shape)
            return A_subset, B_subset, unique_indices_A, unique_indices_B

        def chamfer_distance(x, y):

            # Compute pairwise distances
            x_expanded = x.unsqueeze(2)  # Shape: (B, N, 1, D)
            y_expanded = y.unsqueeze(1)  # Shape: (B, 1, M, D)
            
            # L2 distance
            distances = torch.norm(x_expanded - y_expanded, p=2, dim=-1)  # Shape: (B, N, M)

            # For each point in x, find the nearest neighbor in y
            min_dist_x_to_y, _ = torch.min(distances, dim=2)  # Shape: (B, N)
            
            # For each point in y, find the nearest neighbor in x
            min_dist_y_to_x, _ = torch.min(distances, dim=1)  # Shape: (B, M)
            
            # Average the minimum distances
            loss = torch.mean(min_dist_x_to_y) + torch.mean(min_dist_y_to_x)
            return loss

        frame_num = x.shape[0]

        human_vertices = data["human_vertices"] # frame_num x M x 3
        object_vertices = data["object_vertices"] # N x 3

        batch_obj_vertices = object_vertices.T.unsqueeze(0).repeat(frame_num, 1, 1)
        obj_R = get_rot_matrix(x[:, :6], 'rot_matrix') # frame_num x 3 x 3
        obj_T = x[:, 6:9].reshape((frame_num, 3)).unsqueeze(2) # frame_num x 3 x 1

        transformed_obj_vertices = torch.bmm(obj_R, batch_obj_vertices) + obj_T # frame_num x 3 x N + frame_num x 3 x 1
        _, __, human_contact_indices, object_contact_indices = find_close_points(human_vertices[0], transformed_obj_vertices[0].T)

        import trimesh

        obj_pcd = trimesh.points.PointCloud(transformed_obj_vertices.detach().cpu().numpy()[0].T)
        obj_pcd.export('obj_pcd.ply')
        human_pcd = trimesh.points.PointCloud(human_vertices.detach().cpu().numpy()[0])
        human_pcd.export('human_pcd.ply')


        if len(human_contact_indices) > 0 and len(object_contact_indices) > 0:
            contact_loss = chamfer_distance(human_vertices[:, human_contact_indices, :], torch.transpose(transformed_obj_vertices, 1, 2)[:, object_contact_indices, :])
            print(contact_loss)
        else:
            contact_loss = 0

        return contact_loss * 100
    
    def score_eval_wrapper(data):
        """A wrapper of the score-based model for use by the ODE solver."""
        with torch.no_grad():
            score = score_model(data)
        
        # with torch.enable_grad():
        #     t = data['timestep']
        #     sampled_x = data['sampled_pose']

        #     drift, diffusion = sde_coeff(torch.tensor(t))
        #     drift = drift - diffusion**2*score       # R-SDE
        #     recovered_x = sampled_x + drift * ((1 - t)/(1000 if num_steps is None else num_steps))
        #     recovered_x.requires_grad_(True)

        #     sm_loss = smoothness_loss(recovered_x)
        #     sm_grad = torch.autograd.grad(-sm_loss, recovered_x)[0]
        #     # print(f"score: {score.cpu().numpy().shape}, sm_grad: {sm_grad.cpu().numpy().shape}")
        #     recovered_x.requires_grad_(False)
        #     print(score.cpu().numpy().reshape((-1,)).max())
        #     print(score.cpu().numpy().reshape((-1,)).min())
        #     print(sm_grad.cpu().numpy().reshape((-1,)).max())
        #     print(sm_grad.cpu().numpy().reshape((-1,)).min())

        # return score.cpu().numpy().reshape((-1,)) + sm_grad.cpu().numpy().reshape((-1,))
        return score.cpu().numpy().reshape((-1,)) 
    
    def ode_func(t, x):      
        """The ODE function for use by the ODE solver."""
        x = torch.tensor(x.reshape(-1, pose_dim), dtype=torch.float32, device=device)
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        drift, diffusion = sde_coeff(torch.tensor(t))
        drift = drift.cpu().numpy()
        diffusion = diffusion.cpu().numpy()
        data['sampled_pose'] = x
        data['t'] = time_steps
        data["timestep"] = t ## CUSTOM

        with torch.no_grad():
            score = score_model(data)
        
        # with torch.enable_grad():
        #     # t = data['timestep']
        #     # sampled_x = data['sampled_pose']

        #     drift_, diffusion_ = sde_coeff(torch.tensor(t))
        #     drift_ = drift_- diffusion_**2*score       # R-SDE
        #     recovered_x = x + drift_ * ((1 - t)/(1000 if num_steps is None else num_steps))

        #     recovered_x.requires_grad_(True)
        #     sm_loss = smoothness_loss(recovered_x)
        #     sm_grad = torch.autograd.grad(-sm_loss, recovered_x)[0]
        #     # print(f"score: {score.cpu().numpy().shape}, sm_grad: {sm_grad.cpu().numpy().shape}")

        #     ct_loss = contact_loss(recovered_x)
        #     if ct_loss != 0: ct_grad = torch.autograd.grad(-ct_loss, recovered_x)[0]
        #     else: ct_grad = torch.zeros_like(sm_grad).to('cuda')

        #     recovered_x.requires_grad_(False)

            # print(score.cpu().numpy().reshape((-1,)).max())
            # print(score.cpu().numpy().reshape((-1,)).min())
            # print(sm_grad.cpu().numpy().reshape((-1,)).max())
            # print(sm_grad.cpu().numpy().reshape((-1,)).min())

        # return drift - 0.5 * (diffusion**2) * score_eval_wrapper(data)
        # return drift - 0.5 * (diffusion**2) * score.cpu().numpy().reshape((-1,)) - sm_grad.cpu().numpy().reshape((-1,)) - ct_grad.cpu().numpy().reshape((-1,))
        # return drift - 0.5 * (diffusion**2) * score.cpu().numpy().reshape((-1,)) - sm_grad.cpu().numpy().reshape((-1,))
        return drift - 0.5 * (diffusion**2) * score.cpu().numpy().reshape((-1,))
  
    # Run the black-box ODE solver, note the 
    t_eval = None
    if num_steps is not None:
        # num_steps, from T -> eps
        t_eval = np.linspace(T, eps, num_steps)

    res = integrate.solve_ivp(ode_func, (T, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45', t_eval=t_eval, dense_output=True)
    xs = torch.tensor(res.y, device=device).T.view(-1, batch_size, pose_dim) # [num_steps, bs, pose_dim]
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape) # [bs, pose_dim]
    # denoise, using the predictor step in P-C sampler
    if denoise:
        # Reverse diffusion predictor for denoising
        vec_eps = torch.ones((x.shape[0], 1), device=x.device) * eps
        drift, diffusion = sde_coeff(vec_eps)
        data['sampled_pose'] = x.float()
        data['t'] = vec_eps
        grad = score_model(data)
        drift = drift - diffusion**2*grad       # R-SDE
        mean_x = x + drift * ((1-eps)/(1000 if num_steps is None else num_steps))
        x = mean_x
    
    #num_steps = xs.shape[0]
    #xs = xs.reshape(batch_size*num_steps, -1)
    #xs[:, :6] = normalize_rotation(xs[:, :6], pose_mode)
    #xs[:, 12:18] = normalize_rotation(xs[:, 12:18], pose_mode)
    #xs = xs.reshape(num_steps, batch_size, -1)
    #xs[:, :, -3:] += data['pts_center'].unsqueeze(0).repeat(xs.shape[0], 1, 1)
    #x[:, :6] = normalize_rotation(x[:, :6], pose_mode)
    #x[:, 12:18] = normalize_rotation(x[:, 12:18], pose_mode)
    #x[:, -3:] += data['pts_center']

    return xs.permute(1, 0, 2), x


# def cond_ode_sampler_for_RT_inference(
#         score_model,
#         data,
#         prior,
#         sde_coeff,
#         atol=1e-5, 
#         rtol=1e-5, 
#         device='cuda', 
#         eps=1e-5,
#         T=1.0,
#         num_steps=None,
#         pose_mode='quat_wxyz',
#         denoise=True,
#         init_x=None,
#     ):
#     pose_dim = get_pose_dim_RT_sequence(pose_mode)
#     # batch_size=data['pts'].shape[0]
#     batch_size=data['thetas'].shape[0]
#     contact_threshold=data['contact_threshold']
#     init_x = prior((batch_size, pose_dim), T=T).to(device) if init_x is None else init_x + prior((batch_size, pose_dim), T=T).to(device)
#     shape = init_x.shape
    
#     def smoothness_loss(x):
#         position_smoothness_loss = torch.nn.functional.mse_loss(x[1:, :], x[:-1, :])
#         velocity_smoothness_loss = torch.nn.functional.mse_loss((x[2:, :] - x[1:-1, :]), (x[1:-1, :] - x[:-2, :]))

#         # return (position_smoothness_loss + velocity_smoothness_loss) * 10000
#         return (position_smoothness_loss) * 10000
#         # return (position_smoothness_loss) * 1000
    
#     def contact_loss(x):
#         def find_close_points(A, B, threshold=0.05):
#             A_expanded = A.unsqueeze(1)  # Shape: Nx1x3
#             B_expanded = B.unsqueeze(0)  # Shape: 1xMx3

#             # Compute the squared Euclidean distances between points in A and B
#             distances = torch.norm(A_expanded - B_expanded, dim=2)  # Shape: NxM

#             # Find indices where distance is less than threshold
#             # indices_A, indices_B = torch.where(distances < threshold)
#             # unique_indices_A = torch.unique(indices_A)
#             # unique_indices_B = torch.unique(indices_B)

#             if distances.min() < 1.0:
#                 while True:
#                     indices_A, indices_B = torch.where(distances < threshold)
#                     unique_indices_A = torch.unique(indices_A)
#                     unique_indices_B = torch.unique(indices_B)
#                     if len(unique_indices_A) * len(unique_indices_B) > 1000000:
#                         threshold /= 1.5
#                         continue
#                     break
#             else:
#                 indices_A, indices_B = torch.where(distances < threshold)
#                 unique_indices_A = torch.unique(indices_A)
#                 unique_indices_B = torch.unique(indices_B)

#             A_subset = A[unique_indices_A]
#             B_subset = B[unique_indices_B]

#             # print(unique_indices_A.shape)
#             # print(unique_indices_B.shape)
#             return A_subset, B_subset, unique_indices_A[:], unique_indices_B[:]

#         def chamfer_distance(x, y):

#             # Compute pairwise distances
#             x_expanded = x.unsqueeze(2)  # Shape: (B, N, 1, D)
#             y_expanded = y.unsqueeze(1)  # Shape: (B, 1, M, D)
            
#             # L2 distance
#             distances = torch.norm(x_expanded - y_expanded, p=2, dim=-1)  # Shape: (B, N, M)

#             # For each point in x, find the nearest neighbor in y
#             min_dist_x_to_y, _ = torch.min(distances, dim=2)  # Shape: (B, N)
            
#             # For each point in y, find the nearest neighbor in x
#             min_dist_y_to_x, _ = torch.min(distances, dim=1)  # Shape: (B, M)
            
#             # Average the minimum distances
#             loss = torch.mean(min_dist_x_to_y) + torch.mean(min_dist_y_to_x)

#             minimum_distance = torch.min(min_dist_x_to_y)

#             return loss, torch.mean(min_dist_y_to_x)
        
#         minimum_threshold_distance = contact_threshold

#         frame_num = x.shape[0]

#         human_vertices = data["human_vertices"] # frame_num x M x 3
#         object_vertices = data["object_vertices"] # N x 3
#         ratio = data["ratio"]
        

#         if len(object_vertices.shape) == 2:
#             batch_obj_vertices = object_vertices.T.unsqueeze(0).repeat(frame_num, 1, 1) # frame_num x 3 x N
#         elif len(object_vertices.shape) == 3:
#             batch_obj_vertices = object_vertices.transpose(1, 2) # frame_num x 3 x N

#         obj_R = get_rot_matrix(x[:, :6], 'rot_matrix') # frame_num x 3 x 3
#         if ratio is None:
#             obj_T = x[:, 6:9].reshape((frame_num, 3)).unsqueeze(2) # frame_num x 3 x 1
#         else: obj_T = x[:, 6:9].reshape((frame_num, 3)).unsqueeze(2) * ratio # frame_num x 3 x 1

#         transformed_obj_vertices = torch.bmm(obj_R, batch_obj_vertices) + obj_T # frame_num x 3 x N + frame_num x 3 x 1
#         _, __, human_contact_indices, object_contact_indices = find_close_points(human_vertices[0], transformed_obj_vertices[0].T, minimum_threshold_distance)
#         # _, __, human_contact_indices, object_contact_indices = find_close_points_batch(human_vertices, transformed_obj_vertices.transpose(1, 2), minimum_threshold_distance)

#         # import trimesh
#         # obj_pcd = trimesh.points.PointCloud(transformed_obj_vertices.detach().cpu().numpy()[0].T)
#         # obj_pcd.export('obj_pcd.ply')
#         # human_pcd = trimesh.points.PointCloud(human_vertices.detach().cpu().numpy()[0])
#         # human_pcd.export('human_pcd.ply')

#         minimum_distance = np.inf
#         if len(human_contact_indices) > 0 and len(object_contact_indices) > 0:
#             contact_loss, minimum_distance = chamfer_distance(human_vertices[:, human_contact_indices, :], torch.transpose(transformed_obj_vertices, 1, 2)[:, object_contact_indices, :])
#             # minimum_threshold_distance = minimum_distance * 1.05
#             # print(contact_loss)
#             # print(minimum_distance)
#         else:
#             contact_loss = 0

#         # return max(contact_loss - 0.04, 0.0) * min(70, minimum_threshold_distance * 10000)
#         return minimum_distance, contact_loss * 120
    
#     def contact_loss_qual(x):
#         def find_close_points(A, B, threshold=0.05):
#             A_expanded = A.unsqueeze(1)  # Shape: Nx1x3
#             B_expanded = B.unsqueeze(0)  # Shape: 1xMx3

#             distances = torch.norm(A_expanded - B_expanded, dim=2)  # Shape: NxM

#             indices_A, indices_B = torch.where(distances < threshold)
#             unique_indices_A = torch.unique(indices_A)
#             unique_indices_B = torch.unique(indices_B)

#             # if distances.min() < 1.0:
#             #     while True:
#             #         indices_A, indices_B = torch.where(distances < threshold)
#             #         unique_indices_A = torch.unique(indices_A)
#             #         unique_indices_B = torch.unique(indices_B)
#             #         if len(unique_indices_A) * len(unique_indices_B) > 500000:
#             #             threshold *= 0.75
#             #             continue
#             #         break
#             # else:
#             #     indices_A, indices_B = torch.where(distances < threshold)
#             #     unique_indices_A = torch.unique(indices_A)
#             #     unique_indices_B = torch.unique(indices_B)

#             A_subset = A[unique_indices_A]
#             B_subset = B[unique_indices_B]

#             return A_subset, B_subset, unique_indices_A[:], unique_indices_B[:]

#         def chamfer_distance(x, y):

#             # Compute pairwise distances
#             x_expanded = x.unsqueeze(2)  # Shape: (B, N, 1, D)
#             y_expanded = y.unsqueeze(1)  # Shape: (B, 1, M, D)
            
#             # L2 distance
#             distances = torch.norm(x_expanded - y_expanded, p=2, dim=-1)  # Shape: (B, N, M)

#             # For each point in x, find the nearest neighbor in y
#             min_dist_x_to_y, _ = torch.min(distances, dim=2)  # Shape: (B, N)
            
#             # For each point in y, find the nearest neighbor in x
#             min_dist_y_to_x, _ = torch.min(distances, dim=1)  # Shape: (B, M)
            
#             # Average the minimum distances
#             loss = torch.mean(min_dist_x_to_y) + torch.mean(min_dist_y_to_x)

#             minimum_distance = torch.min(min_dist_x_to_y)

#             return loss, torch.mean(min_dist_y_to_x)
        
#         minimum_threshold_distance = 0.06
#         frame_num = x.shape[0]

#         human_vertices = data["human_vertices"] # frame_num x M x 3
#         object_vertices = data["object_vertices"] # N x 3
#         ratio = data["ratio"]
        

#         if len(object_vertices.shape) == 2:
#             batch_obj_vertices = object_vertices.T.unsqueeze(0).repeat(frame_num, 1, 1) # frame_num x 3 x N
#         elif len(object_vertices.shape) == 3:
#             batch_obj_vertices = object_vertices.transpose(1, 2) # frame_num x 3 x N

#         obj_R = get_rot_matrix(x[:, :6], 'rot_matrix') # frame_num x 3 x 3
#         if ratio is None:
#             obj_T = x[:, 6:9].reshape((frame_num, 3)).unsqueeze(2) # frame_num x 3 x 1
#         else: obj_T = x[:, 6:9].reshape((frame_num, 3)).unsqueeze(2) * ratio # frame_num x 3 x 1

#         transformed_obj_vertices = torch.bmm(obj_R, batch_obj_vertices) + obj_T # frame_num x 3 x N + frame_num x 3 x 1
#         _, __, human_contact_indices, object_contact_indices = find_close_points(human_vertices[0], transformed_obj_vertices[0].T, minimum_threshold_distance)

#         minimum_distance = np.inf
#         if len(human_contact_indices) > 0 and len(object_contact_indices) > 0:
#             contact_loss, minimum_distance = chamfer_distance(human_vertices[:, human_contact_indices, :], torch.transpose(transformed_obj_vertices, 1, 2)[:, object_contact_indices, :])
#             # contact_loss2, minimum_distance = chamfer_distance(human_vertices[0, human_contact_indices, :].unsqueeze(0), torch.transpose(transformed_obj_vertices, 1, 2)[0, object_contact_indices, :].unsqueeze(0))
#             print(minimum_distance)
#         else:
#             contact_loss = 0
#             contact_loss2 = 0

#         # return max(contact_loss - 0.04, 0.0) * min(70, minimum_threshold_distance * 10000)
#         # return (contact_loss + contact_loss2) * 10
#         return minimum_distance, contact_loss * 100
    
#     def score_eval_wrapper(data):
#         """A wrapper of the score-based model for use by the ODE solver."""
#         with torch.no_grad():
#             score = score_model(data)
        
#         return score.cpu().numpy().reshape((-1,)) 
    
#     def ode_func(t, x, start_time, max_duration):      
#         """The ODE function for use by the ODE solver."""
#         x = torch.tensor(x.reshape(-1, pose_dim), dtype=torch.float32, device=device)
#         time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
#         drift, diffusion = sde_coeff(torch.tensor(t))
#         drift = drift.cpu().numpy()
#         diffusion = diffusion.cpu().numpy()
#         data['sampled_pose'] = x
#         data['t'] = time_steps
#         data["timestep"] = t ## CUSTOM

#         with torch.no_grad():
#             score = score_model(data)
        
#         with torch.enable_grad():
#             # t = data['timestep']
#             # sampled_x = data['sampled_pose']

#             drift_, diffusion_ = sde_coeff(torch.tensor(t))
#             drift_ = drift_- diffusion_**2*score       # R-SDE
#             recovered_x = x + drift_ * ((1 - t)/(1000 if num_steps is None else num_steps))

#             recovered_x.requires_grad_(True)
#             sm_loss = smoothness_loss(recovered_x)
#             sm_grad = torch.autograd.grad(-sm_loss, recovered_x)[0]

#             min_dist, ct_loss = contact_loss(recovered_x)
#             # min_dist, ct_loss = contact_loss_qual(recovered_x)
#             if ct_loss != 0: ct_grad = torch.autograd.grad(-ct_loss, recovered_x)[0]
#             else: ct_grad = torch.zeros_like(sm_grad).to('cuda')

#             recovered_x.requires_grad_(False)

#             # print(score.cpu().numpy().reshape((-1,)).max())
#             # print(score.cpu().numpy().reshape((-1,)).min())
#             # print(sm_grad.cpu().numpy().reshape((-1,)).max())
#             # print(sm_grad.cpu().numpy().reshape((-1,)).min())

#         # return drift - 0.5 * (diffusion**2) * score_eval_wrapper(data)
#         return drift - 0.5 * (diffusion**2) * score.cpu().numpy().reshape((-1,)) - sm_grad.cpu().numpy().reshape((-1,)) - ct_grad.cpu().numpy().reshape((-1,))
#         # return drift - 0.5 * (diffusion**2) * score.cpu().numpy().reshape((-1,)) - ct_grad.cpu().numpy().reshape((-1,))
#         # return drift - 0.5 * (diffusion**2) * score.cpu().numpy().reshape((-1,)) - sm_grad.cpu().numpy().reshape((-1,))
#         # return drift - 0.5 * (diffusion**2) * score.cpu().numpy().reshape((-1,))
  
#     # Run the black-box ODE solver, note the 
#     t_eval = None
#     if num_steps is not None:
#         # num_steps, from T -> eps
#         t_eval = np.linspace(T, eps, num_steps)

#     import time
#     def time_limit_event(t, y, start_time, max_duration):
#         print(t)
#         return 1.0
        

#     start_time = time.time()
#     max_duration = 180
#     time_limit_event.terminal = True
#     time_limit_event.direction = -0.01

#     res = integrate.solve_ivp(ode_func, (T, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45', t_eval=t_eval, dense_output=True, events=(time_limit_event,), args=(start_time, max_duration), max_step=1)
#     xs = torch.tensor(res.y, device=device).T.view(-1, batch_size, pose_dim) # [num_steps, bs, pose_dim]
#     x = torch.tensor(res.y[:, -1], device=device).reshape(shape) # [bs, pose_dim]
#     # denoise, using the predictor step in P-C sampler
#     if denoise:
#         # Reverse diffusion predictor for denoising
#         vec_eps = torch.ones((x.shape[0], 1), device=x.device) * eps
#         drift, diffusion = sde_coeff(vec_eps)
#         data['sampled_pose'] = x.float()
#         data['t'] = vec_eps
#         grad = score_model(data)
#         drift = drift - diffusion**2*grad       # R-SDE
#         mean_x = x + drift * ((1-eps)/(1000 if num_steps is None else num_steps))
#         x = mean_x
    
#     #num_steps = xs.shape[0]
#     #xs = xs.reshape(batch_size*num_steps, -1)
#     #xs[:, :6] = normalize_rotation(xs[:, :6], pose_mode)
#     #xs[:, 12:18] = normalize_rotation(xs[:, 12:18], pose_mode)
#     #xs = xs.reshape(num_steps, batch_size, -1)
#     #xs[:, :, -3:] += data['pts_center'].unsqueeze(0).repeat(xs.shape[0], 1, 1)
#     #x[:, :6] = normalize_rotation(x[:, :6], pose_mode)
#     #x[:, 12:18] = normalize_rotation(x[:, 12:18], pose_mode)
#     #x[:, -3:] += data['pts_center']

#     return xs.permute(1, 0, 2), x



def cond_ode_sampler_for_RT_inference(
        score_model,
        data,
        prior,
        sde_coeff,
        atol=1e-5, 
        rtol=1e-5, 
        device='cuda', 
        eps=1e-5,
        T=1.0,
        num_steps=None,
        pose_mode='quat_wxyz',
        denoise=True,
        init_x=None,
    ):
    pose_dim = get_pose_dim_RT_sequence(pose_mode)
    # batch_size=data['pts'].shape[0]
    batch_size=data['thetas'].shape[0]
    init_x = prior((batch_size, pose_dim), T=T).to(device) if init_x is None else init_x + prior((batch_size, pose_dim), T=T).to(device)
    shape = init_x.shape
    
    def smoothness_loss(x):
        position_smoothness_loss = torch.nn.functional.mse_loss(x[1:, :], x[:-1, :])
        velocity_smoothness_loss = torch.nn.functional.mse_loss((x[2:, :] - x[1:-1, :]), (x[1:-1, :] - x[:-2, :]))

        # return (position_smoothness_loss + velocity_smoothness_loss) * 10000
        return (position_smoothness_loss) * 10000
    
    def contact_loss(x):
        def find_close_points(A, B, threshold=0.1):
            A_expanded = A.unsqueeze(1)  # Shape: Nx1x3
            B_expanded = B.unsqueeze(0)  # Shape: 1xMx3

            # Compute the squared Euclidean distances between points in A and B
            distances = torch.norm(A_expanded - B_expanded, dim=2)  # Shape: NxM

            # Find indices where distance is less than threshold
            indices_A, indices_B = torch.where(distances < threshold)
            unique_indices_A = torch.unique(indices_A)
            unique_indices_B = torch.unique(indices_B)

            A_subset = A[unique_indices_A]
            B_subset = B[unique_indices_B]

            print(unique_indices_A.shape)
            print(unique_indices_B.shape)
            return A_subset, B_subset, unique_indices_A, unique_indices_B

        def chamfer_distance(x, y):

            # Compute pairwise distances
            x_expanded = x.unsqueeze(2)  # Shape: (B, N, 1, D)
            y_expanded = y.unsqueeze(1)  # Shape: (B, 1, M, D)
            
            # L2 distance
            distances = torch.norm(x_expanded - y_expanded, p=2, dim=-1)  # Shape: (B, N, M)

            # For each point in x, find the nearest neighbor in y
            min_dist_x_to_y, _ = torch.min(distances, dim=2)  # Shape: (B, N)
            
            # For each point in y, find the nearest neighbor in x
            min_dist_y_to_x, _ = torch.min(distances, dim=1)  # Shape: (B, M)
            
            # Average the minimum distances
            loss = torch.mean(min_dist_x_to_y) + torch.mean(min_dist_y_to_x)
            return loss

        frame_num = x.shape[0]

        human_vertices = data["human_vertices"] # frame_num x M x 3
        object_vertices = data["object_vertices"] # N x 3

        batch_obj_vertices = object_vertices.T.unsqueeze(0).repeat(frame_num, 1, 1)
        obj_R = get_rot_matrix(x[:, :6], 'rot_matrix') # frame_num x 3 x 3
        obj_T = x[:, 6:9].reshape((frame_num, 3)).unsqueeze(2) # frame_num x 3 x 1

        transformed_obj_vertices = torch.bmm(obj_R, batch_obj_vertices) + obj_T # frame_num x 3 x N + frame_num x 3 x 1
        _, __, human_contact_indices, object_contact_indices = find_close_points(human_vertices[0], transformed_obj_vertices[0].T)

        import trimesh

        obj_pcd = trimesh.points.PointCloud(transformed_obj_vertices.detach().cpu().numpy()[0].T)
        obj_pcd.export('obj_pcd.ply')
        human_pcd = trimesh.points.PointCloud(human_vertices.detach().cpu().numpy()[0])
        human_pcd.export('human_pcd.ply')


        if len(human_contact_indices) > 0 and len(object_contact_indices) > 0:
            contact_loss = chamfer_distance(human_vertices[:, human_contact_indices, :], torch.transpose(transformed_obj_vertices, 1, 2)[:, object_contact_indices, :])
            print(contact_loss)
        else:
            contact_loss = 0

        return contact_loss * 100
    
    def score_eval_wrapper(data):
        """A wrapper of the score-based model for use by the ODE solver."""
        with torch.no_grad():
            score = score_model(data)
        
        # with torch.enable_grad():
        #     t = data['timestep']
        #     sampled_x = data['sampled_pose']

        #     drift, diffusion = sde_coeff(torch.tensor(t))
        #     drift = drift - diffusion**2*score       # R-SDE
        #     recovered_x = sampled_x + drift * ((1 - t)/(1000 if num_steps is None else num_steps))
        #     recovered_x.requires_grad_(True)

        #     sm_loss = smoothness_loss(recovered_x)
        #     sm_grad = torch.autograd.grad(-sm_loss, recovered_x)[0]
        #     # print(f"score: {score.cpu().numpy().shape}, sm_grad: {sm_grad.cpu().numpy().shape}")
        #     recovered_x.requires_grad_(False)
        #     print(score.cpu().numpy().reshape((-1,)).max())
        #     print(score.cpu().numpy().reshape((-1,)).min())
        #     print(sm_grad.cpu().numpy().reshape((-1,)).max())
        #     print(sm_grad.cpu().numpy().reshape((-1,)).min())

        # return score.cpu().numpy().reshape((-1,)) + sm_grad.cpu().numpy().reshape((-1,))
        return score.cpu().numpy().reshape((-1,)) 
    
    def ode_func(t, x):      
        """The ODE function for use by the ODE solver."""
        x = torch.tensor(x.reshape(-1, pose_dim), dtype=torch.float32, device=device)
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        drift, diffusion = sde_coeff(torch.tensor(t))
        drift = drift.cpu().numpy()
        diffusion = diffusion.cpu().numpy()
        data['sampled_pose'] = x
        data['t'] = time_steps
        data["timestep"] = t ## CUSTOM

        with torch.no_grad():
            score = score_model(data)
        
        # with torch.enable_grad():
        #     # t = data['timestep']
        #     # sampled_x = data['sampled_pose']

        #     drift_, diffusion_ = sde_coeff(torch.tensor(t))
        #     drift_ = drift_- diffusion_**2*score       # R-SDE
        #     recovered_x = x + drift_ * ((1 - t)/(1000 if num_steps is None else num_steps))

        #     recovered_x.requires_grad_(True)
        #     sm_loss = smoothness_loss(recovered_x)
        #     sm_grad = torch.autograd.grad(-sm_loss, recovered_x)[0]
        #     # print(f"score: {score.cpu().numpy().shape}, sm_grad: {sm_grad.cpu().numpy().shape}")

        #     ct_loss = contact_loss(recovered_x)
        #     if ct_loss != 0: ct_grad = torch.autograd.grad(-ct_loss, recovered_x)[0]
        #     else: ct_grad = torch.zeros_like(sm_grad).to('cuda')

        #     recovered_x.requires_grad_(False)

            # print(score.cpu().numpy().reshape((-1,)).max())
            # print(score.cpu().numpy().reshape((-1,)).min())
            # print(sm_grad.cpu().numpy().reshape((-1,)).max())
            # print(sm_grad.cpu().numpy().reshape((-1,)).min())

        # return drift - 0.5 * (diffusion**2) * score_eval_wrapper(data)
        # return drift - 0.5 * (diffusion**2) * score.cpu().numpy().reshape((-1,)) - sm_grad.cpu().numpy().reshape((-1,)) - ct_grad.cpu().numpy().reshape((-1,))
        # return drift - 0.5 * (diffusion**2) * score.cpu().numpy().reshape((-1,)) - sm_grad.cpu().numpy().reshape((-1,))
        return drift - 0.5 * (diffusion**2) * score.cpu().numpy().reshape((-1,))
  
    # Run the black-box ODE solver, note the 
    t_eval = None
    if num_steps is not None:
        # num_steps, from T -> eps
        t_eval = np.linspace(T, eps, num_steps)

    res = integrate.solve_ivp(ode_func, (T, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45', t_eval=t_eval, dense_output=True)
    xs = torch.tensor(res.y, device=device).T.view(-1, batch_size, pose_dim) # [num_steps, bs, pose_dim]
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape) # [bs, pose_dim]
    # denoise, using the predictor step in P-C sampler
    if denoise:
        # Reverse diffusion predictor for denoising
        vec_eps = torch.ones((x.shape[0], 1), device=x.device) * eps
        drift, diffusion = sde_coeff(vec_eps)
        data['sampled_pose'] = x.float()
        data['t'] = vec_eps
        grad = score_model(data)
        drift = drift - diffusion**2*grad       # R-SDE
        mean_x = x + drift * ((1-eps)/(1000 if num_steps is None else num_steps))
        x = mean_x
    
    #num_steps = xs.shape[0]
    #xs = xs.reshape(batch_size*num_steps, -1)
    #xs[:, :6] = normalize_rotation(xs[:, :6], pose_mode)
    #xs[:, 12:18] = normalize_rotation(xs[:, 12:18], pose_mode)
    #xs = xs.reshape(num_steps, batch_size, -1)
    #xs[:, :, -3:] += data['pts_center'].unsqueeze(0).repeat(xs.shape[0], 1, 1)
    #x[:, :6] = normalize_rotation(x[:, :6], pose_mode)
    #x[:, 12:18] = normalize_rotation(x[:, 12:18], pose_mode)
    #x[:, -3:] += data['pts_center']

    return xs.permute(1, 0, 2), x

def cond_edm_sampler(
    decoder_model, data, prior_fn, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    pose_mode='quat_wxyz', device='cuda'
):
    pose_dim = get_pose_dim_RT_sequence(pose_mode)
    batch_size = data['pts'].shape[0]
    latents = prior_fn((batch_size, pose_dim)).to(device)

    # Time step discretization. note that sigma and t is interchangable
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    def decoder_wrapper(decoder, data, x, t):
        # save temp
        x_, t_= data['sampled_pose'], data['t']
        # init data
        data['sampled_pose'], data['t'] = x, t
        # denoise
        data, denoised = decoder(data)
        # recover data
        data['sampled_pose'], data['t'] = x_, t_
        return denoised.to(torch.float64)

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    xs = []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = torch.as_tensor(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = decoder_wrapper(decoder_model, data, x_hat, t_hat)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = decoder_wrapper(decoder_model, data, x_next, t_next)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        xs.append(x_next.unsqueeze(0))

    xs = torch.stack(xs, dim=0) # [num_steps, bs, pose_dim]
    x = xs[-1] # [bs, pose_dim]

    # post-processing
    xs = xs.reshape(batch_size*num_steps, -1)
    xs[:, :-3] = normalize_rotation(xs[:, :-3], pose_mode)
    xs = xs.reshape(num_steps, batch_size, -1)
    xs[:, :, -3:] += data['pts_center'].unsqueeze(0).repeat(xs.shape[0], 1, 1)
    x[:, :-3] = normalize_rotation(x[:, :-3], pose_mode)
    x[:, -3:] += data['pts_center']

    return xs.permute(1, 0, 2), x


