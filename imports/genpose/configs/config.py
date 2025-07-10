import argparse
from ipdb import set_trace
from k import k

def get_config():
    parser = argparse.ArgumentParser()
    
    """ dataset """
    parser.add_argument('--dataset', type=str, default='ComAsset')
    parser.add_argument('--category', type=str, default='frypan')
    parser.add_argument('--hoi_data_dir', type=str, default='results/david/hoi_data')
    parser.add_argument('--scale_dir', type=str, default='results/generation/scale')
    parser.add_argument('--hmr_dir', type=str, default='results/generation/hmr')
    parser.add_argument('--omdm_dir', type=str, default='results/david/omdm')
    parser.add_argument('--camera_dir', type=str, default='results/generation/cameras')
    parser.add_argument('--inference_epoch', type=int, default=2000)
    parser.add_argument('--inference_human_motion_dir', type=str, default='results/inference/human_motion')
    parser.add_argument('--inference_object_motion_dir', type=str, default='results/inference/object_motion')
    parser.add_argument('--inference_contact_threshold', type=float, default=0.05)
    parser.add_argument('--inference_object_scale', type=float, default=None)
    parser.add_argument('--skip_done', action='store_true', default=False)


    parser.add_argument('--synset_names', nargs='+', default=['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug'])
    parser.add_argument('--selected_classes', nargs='+')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--o2c_pose', default=True, action='store_true')
    parser.add_argument('--batch_size', type=int, default=192)
    parser.add_argument('--max_batch_size', type=int, default=192)
    parser.add_argument('--mini_bs', type=int, default=192)
    parser.add_argument('--pose_mode', type=str, default='rot_matrix')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--percentage_data_for_train', type=float, default=1.0) 
    parser.add_argument('--percentage_data_for_val', type=float, default=1.0) 
    parser.add_argument('--percentage_data_for_test', type=float, default=1.0) 
    parser.add_argument('--train_source', type=str, default='CAMERA+Real')
    parser.add_argument('--val_source', type=str, default='CAMERA')
    parser.add_argument('--test_source', type=str, default='Real')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--per_obj', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=32)
    
    
    """ model """
    parser.add_argument('--posenet_mode',  type=str, default='score')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--sampler_mode', nargs='+', default=['ode'])
    parser.add_argument('--sampling_steps', type=int, default=500)
    parser.add_argument('--sde_mode', type=str, default='ve')
    parser.add_argument('--sigma', type=float, default=25) # base-sigma for SDE
    parser.add_argument('--likelihood_weighting', default=False, action='store_true')
    parser.add_argument('--regression_head', type=str, default='Rx_Ry_and_T')
    parser.add_argument('--pointnet2_params', type=str, default='light')
    parser.add_argument('--pts_encoder', type=str, default='pointnet2') 
    parser.add_argument('--energy_mode', type=str, default='IP') 
    parser.add_argument('--s_theta_mode', type=str, default='score') 
    parser.add_argument('--norm_energy', type=str, default='identical')
    
    
    """ training """
    parser.add_argument('--agent_type', type=str, default='score', help='one of the [score, energy, energy_with_ranking]')
    parser.add_argument('--pretrained_score_model_path', type=str)
    parser.add_argument('--pretrained_energy_model_path', type=str)
    parser.add_argument('--distillation', default=False, action='store_true')
    parser.add_argument('--n_epochs', type=int, default=10000)
    parser.add_argument('--log_dir', type=str, default='debug')
    parser.add_argument('--optimizer',  type=str, default='Adam')
    parser.add_argument('--eval_freq', type=int, default=100)
    parser.add_argument('--repeat_num', type=int, default=20)
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--ema_rate', type=float, default=0.999)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--lr_decay', type=float, default=0.99)
    parser.add_argument('--use_pretrain', default=False, action='store_true')
    parser.add_argument('--parallel', default=False, action='store_true')   
    parser.add_argument('--num_gpu', type=int, default=4)
    parser.add_argument('--is_train', default=True, action='store_true')
    
    
    """ testing """
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--pred', default=False, action='store_true')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--eval_repeat_num', type=int, default=50)
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--max_eval_num', type=int, default=10000000)
    parser.add_argument('--results_path', type=str, default='')
    parser.add_argument('--T0', type=float, default=1.0)
    
    
    """ nocs_mrcnn testing"""
    parser.add_argument('--img_size', type=int, default=256, help='cropped image size')
    parser.add_argument('--result_dir', type=str, default='', help='result directory')
    parser.add_argument('--model_dir_list', nargs='+')
    parser.add_argument('--energy_model_dir', type=str, default='', help='energy network ckpt directory')
    parser.add_argument('--score_model_dir', type=str, default='', help='score network ckpt directory')
    parser.add_argument('--ranker', type=str, default='energy_ranker', help='energy_ranker, gt_ranker or random')
    parser.add_argument('--pooling_mode', type=str, default='nearest', help='nearest or average')
    
    parser.add_argument('--score_model_path', type=str, default='', help='score network ckpt path')

    parser.add_argument('--skip_num', type=int, default=0)

    cfg = parser.parse_args()
    
    # cfg.data_path = f"{cfg.hoi_data_dir}/{cfg.dataset}/{cfg.category}"
    # cfg.log_dir = f"{cfg.omdm_dir}/{cfg.dataset}/{cfg.category}_k{k}"
    # cfg.score_model_path = f"{cfg.omdm_dir}/{cfg.dataset}/{cfg.category}_k{k}/ckpt_epoch{cfg.inference_epoch}.pth"

    cfg.data_path = f"{cfg.hoi_data_dir}/{cfg.dataset}/{cfg.category}"
    cfg.log_dir = f"{cfg.omdm_dir}/{cfg.dataset}/{cfg.category}"
    cfg.score_model_path = f"{cfg.omdm_dir}/{cfg.dataset}/{cfg.category}/ckpt_epoch{cfg.inference_epoch}.pth"


    # dynamic zoom in parameters
    cfg.DYNAMIC_ZOOM_IN_PARAMS = {
        'DZI_PAD_SCALE': 1.5,
        'DZI_TYPE': 'uniform',
        'DZI_SCALE_RATIO': 0.25,
        'DZI_SHIFT_RATIO': 0.25
    }
    
    # pts aug parameters
    cfg.PTS_AUG_PARAMS = {
        'aug_pc_pro': 0.2,
        'aug_pc_r': 0.2,
        'aug_rt_pro': 0.3,
        'aug_bb_pro': 0.3,
        'aug_bc_pro': 0.3
    }
    
    # 2D aug parameters
    cfg.DEFORM_2D_PARAMS = {
        'roi_mask_r': 3,
        'roi_mask_pro': 0.5
    }
    
    return cfg

