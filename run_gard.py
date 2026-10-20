import os
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
from functools import reduce
import imageio.v2 as imageio
import torch
from torch import nn, optim
import torch.nn.functional as F

# 导入自定义工具模块
from ms_utils_gard import load_decoder_and_message, bit_accuracy, lpips
from attack_utils import Attacker
from dnerf_utils import load_4dgs_scene, get_valid_cameras

# 4DGS core imports
import sys
sys.path.append(os.path.join(os.getcwd(), '4DGaussians'))
from arguments import ModelParams, PipelineParams, ModelHiddenParams, OptimizationParams, get_combined_args
from gaussian_renderer import render

from utils.image_utils import psnr
from utils.loss_utils import l1_loss, l2_loss, ssim
from utils.general_utils import safe_state
from utils.params_utils import merge_hparams

SH_KEYS = ['features_dc', 'features_rest']

# 渲染
def render_views(gaussians, cameras, renderArgs, cam_type=None, show_progress=True):
    """
    在 4D Gaussian Splatting 上渲染多个相机视角。

    Args:
        gaussians: GaussianModel 对象，包含训练好的点云模型
        cameras: list[Camera]，相机视角列表（来自 Scene.getVideoCameras() 等）
        renderArgs: tuple(pipe, background)
            - pipe: PipelineParams 对象
            - background: torch.Tensor，背景颜色 (cuda, shape=[3])
        cam_type: (可选) 数据集类型（Blender, DyNeRF, Nerfies, PanopticSports）
        show_progress (bool): 是否显示进度条

    Returns:
        torch.Tensor: [N, 3, H, W]，渲染结果，像素范围 [0,1]
    """
    pipe, background = renderArgs
    videos = []
    for i in range(len(cameras)):
        video = cameras[i]
        video_images = []
        for frame in video:
            out = render(frame, gaussians, pipe, background, cam_type=cam_type)
            img = torch.clamp(out["render"], 0.0, 1.0)
            video_images.append(img)
        videos.append(torch.stack(video_images))
    return torch.stack(videos) # [N, F, 3, H, W]
import torch
import torch.nn as nn
import random

class RandomGaussianNoise(nn.Module):
    """
    x' = clamp(x + sigma * N(0,1), 0, 1)

    支持输入:
      - (C,H,W)
      - (B,C,H,W)
      - (A,B,C,H,W) 例如 (views, frames, C, H, W) 或 (B,T,C,H,W)

    5D 情况下：每个 (A,B) 小图像单独采样一个 sigma
    """
    def __init__(self, sigma_min=0.0, sigma_max=0.05, p=1.0, clamp=True):
        super().__init__()
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.p = float(p)
        self.clamp = bool(clamp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p < 1.0 and random.random() > self.p:
            return x

        if x.dim() == 3:  # (C,H,W)
            sigma = (self.sigma_max - self.sigma_min) * torch.rand(
                1, device=x.device, dtype=x.dtype
            ) + self.sigma_min
            y = x + sigma * torch.randn_like(x)
            return y.clamp(0.0, 1.0) if self.clamp else y

        if x.dim() == 4:  # (B,C,H,W)
            B = x.shape[0]
            sigma = (self.sigma_max - self.sigma_min) * torch.rand(
                (B, 1, 1, 1), device=x.device, dtype=x.dtype
            ) + self.sigma_min
            y = x + sigma * torch.randn_like(x)
            return y.clamp(0.0, 1.0) if self.clamp else y

        if x.dim() == 5:  # (A,B,C,H,W)
            A, B = x.shape[0], x.shape[1]
            sigma = (self.sigma_max - self.sigma_min) * torch.rand(
                (A, B, 1, 1, 1), device=x.device, dtype=x.dtype
            ) + self.sigma_min
            y = x + sigma * torch.randn_like(x)
            return y.clamp(0.0, 1.0) if self.clamp else y

        raise ValueError(f"Expect 3D/4D/5D tensor, got {tuple(x.shape)}")

# 水印提取与指标计算
def extract_message(pds, model, args):
    """ 从渲染图像中提取水印消息 """
    messages = [model(pds[idx * args.batch_size:(idx + 1) * args.batch_size]) for idx in range((len(pds) + args.batch_size - 1) // args.batch_size)]
    return torch.cat(messages)

def eval_extraction_accuracy(pds, message, model, args):
    """ 评估水印提取的比特准确率 """
    output = extract_message(pds, model, args)
    target = message.repeat(len(pds), *[1 for _ in range(len(message.shape))])
    # ✅ 确保两者都是布尔值
    output = (torch.sigmoid(output) > 0.5).to(torch.bool)
    target = target.bool()
    return bit_accuracy(output, target)

def eval_extraction_accuracy_under_attacks(pds, message, attacker, model, args):
    """ 攻击下的水印准确率 """
    rets = {}
    for atype in args.atypes:
        torch.cuda.empty_cache()
        attacked_pds = torch.cat([
            model.resample(torch.clamp(attacker(pd, atype)[None], 0, 1)) for pd in pds])
        rets[atype] = eval_extraction_accuracy(attacked_pds, message, model, args)
    return rets

def eval_image_quality(pds, gts, bsize=1):
    new_pds = []
    for pd in pds:
        for i in range(pd.shape[0]):
            new_pds.append(pd[i])
    new_gts = []
    for gt in gts:
        for i in range(gt.shape[0]):
            new_gts.append(gt[i])
    return {
        # 计算峰值信噪比(PSNR)
        'psnr' : reduce(lambda x1, x2 : x1 + x2, [psnr(pd[None], gt[None]).item() for pd, gt in zip(new_pds, new_gts)]) / len(new_pds),
        # 计算结构相似性指数(SSIM)
        'ssim' : reduce(lambda x1, x2 : x1 + x2, [ssim(pd[None], gt[None]).item() for pd, gt in zip(new_pds, new_gts)]) / len(new_pds),
        # 计算感知损失(LPIPS)
        'lpips': reduce(lambda x1, x2 : x1 + x2, [lpips(pd[None], gt[None]).item() for pd, gt in zip(new_pds, new_gts)]) / len(new_pds)
    }


# 训练阶段
def train(dataset, opt, pipe, hyper, args, model, message):
    message_text = reduce(lambda x1, x2 : x1 + x2, [str(x) for x in message.int().cpu().numpy()])
    print (f'message : {message_text}')

    # 创建保存目录与子目录（用于存放所有 .pkl）
    os.makedirs(args.sdir, exist_ok=True)
    ckpt_dir = os.path.join(args.sdir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(args.sdir, 'message.txt'), 'w') as txtfile:
        txtfile.write(message_text)
        print (f"message can be seen in {os.path.join(args.sdir, 'message.txt')}")

    # 初始化4DGS模型和GuardSplat模型
    scene, gaussians, pipe, hyper, info = load_4dgs_scene(args, iteration=args.iteration)
    guardsplat = deepcopy(gaussians)
    
    if args.dataset_type != scene.dataset_type:
        print(f"[WARN] Scene dataset_type = '{scene.dataset_type}' (auto-detected), ignoring manual override '{args.dataset_type}'")
    
    # 获取相机hasattr(TrainCameras, "dataset")
    TrainCameras = scene.getTrainCameras()
    TestCameras = scene.getTestCameras()
    # breakpoint()
    
    # 设置背景颜色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

    # 初始化CUDA事件用于计时
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
   # 渲染真实图像
    with torch.no_grad():
        gt_test = render_views(gaussians, TestCameras, (pipe, background), scene.dataset_type)

    # 初始化球谐函数特征和偏移量
    sh_features = {key: getattr(guardsplat, f'_{key}').clone() for key in SH_KEYS}
    sh_offsets = {key: nn.Parameter(torch.zeros_like(sh_feature).cuda().requires_grad_(True)) for key, sh_feature in sh_features.items()}

    # 设置不同特征的学习率
    params = [{'params' : [sh_offsets[key]], 'lr' : lr} for key, lr in zip(SH_KEYS, [args.learning_rate, args.learning_rate / args.k_rest])]

    ##局部训练model
    # for p in model.msg_visual_enhancer.parameters():
    #     p.requires_grad = True
    # params.append({
    #     'params': model.msg_visual_enhancer.parameters(),
    #     'lr': 1e-3
    # })
    optimizer = optim.Adam(params, lr=0.0, eps=1e-15)
    
    # 初始化攻击器（如果指定了攻击类型）
    train_attacker, test_attacker = [Attacker(args.atypes, flag) for flag in [True, False]] if args.atypes is not None else [None, None]
    
    evaluation(guardsplat, None, TestCameras, test_attacker, (pipe, background), model, message, gt_test, 0, args.eval_interval, args)
    # evaluation_o(guardsplat, None, TestCameras, test_attacker, (pipe, background), model, message, gt_test, 0, args.eval_interval, args)
    # evaluation_n_gts(guardsplat, None, TestCameras, test_attacker, (pipe, background), model, message, gt_test, 0, args.eval_interval, args)
    
    train.prev_sh_offsets = {k: None for k in SH_KEYS}
    # add_noise = RandomGaussianNoise(sigma_min=0.0, sigma_max=0.05, p=0.7)


    # 开始训练循环
    progress_bar = tqdm(range(1, args.num_epochs2 + 1), desc='Training progress')
    for iteration in progress_bar:
        iter_start.record()

        # 初始化损失和消息字典
        ldict, mdict = {k : [] for k in ['recon', 'msg', 'off']}, {k : [] for k in ['output', 'target']}
        used_indice = [i for i in range(len(TrainCameras))]
        
        num_used = args.train_num_views_per_time_real
        

        if args.train_random_sample_view_real:
            random.shuffle(used_indice)
            used_indice = used_indice[:num_used]
        else:
            if num_used >= len(used_indice):
                used_indice = used_indice
            elif num_used <= 0:
                used_indice = []
            else:
                # 在 [0, len(used_indice)-1] 上均匀取 num_used 个位置
                sampled_pos = np.linspace(0, len(used_indice) - 1, num=num_used, dtype=int)
                used_indice = [used_indice[i] for i in sampled_pos]
            
        
        # 按批次处理训练相机
        for iterIdx in tqdm(range((len(used_indice) + args.batch_size - 1) // args.batch_size)):
        # for iterIdx in range((len(TrainCameras) + args.batch_size - 1) // args.batch_size):
            losses = {}
            optimizer.zero_grad(set_to_none=True)
            
            cameras = []
            for used_idx in range(iterIdx * args.batch_size, min((iterIdx + 1) * args.batch_size, len(used_indice))):
                real_used_idx = used_indice[used_idx]
                used_sample = TrainCameras[real_used_idx]
                
                if args.train_random_sample_time_real:
                    if args.train_random_sample_time_real_interval:
                        n = len(used_sample)
                        k = min(args.train_num_frame_real, n)
                        chunks = np.array_split(np.arange(n), k)
                        idx = np.random.randint(min(len(chunk) for chunk in chunks))
                        sampled_pos = [chunk[idx] for chunk in chunks]
                        used_sample = [used_sample[i] for i in sampled_pos]
                    else:
                        k = min(args.train_num_frame_real, len(used_sample))
                        used_sample = random.sample(used_sample, k=k)
                    
                    
                else:
                    n = len(used_sample)
                    k = args.train_num_frame_real
                    sampled_pos = np.linspace(0, n - 1, num=k, dtype=int)
                    used_sample = [used_sample[i] for i in sampled_pos]
                    
                # used_sample = random.sample(used_sample, k=args.train_num_frame_real)
                cameras.append(used_sample)
    
            # 应用球谐函数偏移量
            [setattr(guardsplat, f'_{key}', sh_features[key] + sh_offsets[key]) for key in SH_KEYS]

            # 渲染当前批次的图像
            pds = render_views(guardsplat, cameras, (pipe, background))
            with torch.no_grad():
                gts = render_views(gaussians, cameras, (pipe, background))


            # 计算重建损失
            # rgb_loss = l1_loss(pds, gts) + args.new_lambda_dssim * (1.0 - ssim(pds, gts))
            new_pds = []
            new_gts = []
            for pd in pds:
                for i in range(pd.shape[0]):
                    new_pds.append(pd[i])
            for gt in gts:
                for i in range(gt.shape[0]):
                    new_gts.append(gt[i])
            new_pds = torch.stack(new_pds)
            new_gts = torch.stack(new_gts)
            rgb_loss = (1.0 - args.new_lambda_dssim) * l1_loss(new_pds, new_gts) + args.new_lambda_dssim * (1.0 - ssim(new_pds, new_gts))
            lpips_loss = lpips(new_pds, new_gts) / args.batch_size
            losses['recon'] = rgb_loss + args.new_lambda_lpips * lpips_loss

            # 如果指定了攻击类型，则应用随机攻击
            if args.atypes is not None:
                attacked_pds = []
                for pd in pds:
                    atype = random.choice(['clean', *args.atypes])
                    attacked_pd = pd if atype == 'clean' else train_attacker(pd, atype)
                    attacked_pds.append(model.resample(attacked_pd[None]))
                output = model(torch.cat(attacked_pds))
            else:
                output = model(pds)
                # output_o = model(gts)
                # gts_noisy = add_noise(gts) # gts 是 list，每个 gt 通常是 (B,C,H,W)
                # output_noisy_gts = model(gts_noisy)
                # pds_noisy = add_noise(pds)
                # output_noisy_pds = model(pds_noisy)

            # 准备目标消息
            target = message.repeat(len(output), *[1 for _ in range(len(message.shape))])
            
            # 计算消息损失和偏移量正则化损失
            # o_msg = 1 / (F.binary_cross_entropy_with_logits(output_o, target) + 1e-8)
            # noise_msg = 1 / (F.binary_cross_entropy_with_logits(output_noisy_gts, target) + 1e-8)
            # # attack_msg1 = 1 / (F.binary_cross_entropy_with_logits(output_noisy_pds, target) + 1e-8)
            # neg_msg = (o_msg + noise_msg) / 2
            # # print("o_msg:", o_msg.item(), "noise_msg:", noise_msg.item())
            # losses['msg'] = F.binary_cross_entropy_with_logits(output, target) + neg_msg 
            losses['msg'] = F.binary_cross_entropy_with_logits(output, target)
            losses['off'] = reduce(lambda x1, x2 : x1 + x2, [l2_loss(v, torch.zeros_like(v)) for _, v in sh_offsets.items()])
            
            # # 计算总损失并反向传播
            loss = reduce(lambda x1, x2 : x1 + x2, [getattr(args, f'lambda_{k}') * v for k, v in losses.items()])
            loss.backward()
            optimizer.step()

            # 记录损失值
            for k, v in losses.items():
                ldict[k].append(v.item())

            # 记录消息相关数据
            for k in mdict.keys():
                mdict[k].append(locals()[k])

        # 计算平均损失和比特准确率
        mean = lambda xs : sum(xs) / len(xs)
        ldict = {k : mean(v) for k, v in ldict.items()}
        mdict = {k : torch.cat(v) for k, v in mdict.items()}
        ans = bit_accuracy(**mdict)

        # 格式化日志信息
        logs = reduce(lambda x1, x2 : f'{x1} | {x2}', [f'{k} : {v:.5f}' for k, v in ldict.items()])

        progress_bar.set_description(f'[ITER {iteration}] {logs} | Bit Acc : {ans:.4f}')
        iter_end.record()

        # ✅ 每隔10轮保存checkpoint
        if iteration % 10 == 0:
            ckpt_dir = os.path.join(args.sdir, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            
            save_dict = {
                'message': message,
                **sh_offsets
            }

            ckpt_path = os.path.join(
                ckpt_dir, f'sh_offset_{str(iteration).zfill(len(str(args.num_epochs2)))}.pkl'
            )
            torch.save(save_dict, ckpt_path)
            print(f"[CKPT] Saved checkpoint at {ckpt_path}")
        
        # 定期评估模型性能
        evaluation(guardsplat, sh_offsets, TestCameras, test_attacker, (pipe, background), model, message, gt_test, iteration, args.eval_interval, args)
        # evaluation_o(guardsplat, sh_offsets, TestCameras, test_attacker, (pipe, background), model, message, gt_test, iteration, args.eval_interval, args)
        # evaluation_n_gts(guardsplat, sh_offsets, TestCameras, test_attacker, (pipe, background), model, message, gt_test, iteration, args.eval_interval, args)
        # evaluation_n_pds(guardsplat, sh_offsets, TestCameras, test_attacker, (pipe, background), model, message, gt_test, iteration, args.eval_interval, args)
# 评估函数
@torch.no_grad()
def evaluation(gaussians, sh_offsets, cameras, attacker, renderArgs, model, message, gts, iteration, eval_interval, args):
    """
    评估模型性能
    
    Args:
        gaussians: 高斯模型对象
        sh_offsets: 球谐函数偏移量
        cameras: 相机列表
        attacker: 攻击器对象
        renderArgs: 渲染参数
        model: 水印解码器模型
        message: 水印消息
        gts: 真实图像
        iteration: 当前迭代次数
        eval_interval: 评估间隔
        args: 参数对象
    """
    # 在指定间隔或评估模式下进行评估
    if iteration % eval_interval == 0 or args.mode == 'eval':
        # 渲染当前模型的图像
        pds = render_views(gaussians, cameras, renderArgs)

        # 计算基本评估指标
        ans = {
            'Bit Acc' : eval_extraction_accuracy(pds, message, model, args),
            **eval_image_quality(pds, gts)
        }

        # 如果指定了攻击类型，计算攻击下的准确率
        if args.atypes is not None:
            ans.update(eval_extraction_accuracy_under_attacks(pds, message, attacker, model, args))
            
        # 格式化并打印日志
        logs = reduce(lambda x1, x2 : f'{x1} | {x2}', [f'{k} : {v:.4f}' for k, v in ans.items()])
        print (f'[ITER {iteration}] {logs}' if args.mode == 'train' else logs)

        #save scores
        if args.mode == 'train':
            with open(os.path.join(args.sdir, 'train_scores.log'), 'a') as txtfile:
                txtfile.write(f'[ITER {iteration}] {logs}\n')
        else:
            with open(os.path.join(args.sdir, 'eval_scores.log'), 'a') as txtfile:
                txtfile.write(f'[ITER {iteration}] {logs}\n')

        torch.cuda.empty_cache()
        
        # 在评估模式下保存图像对比
        # if args.save_map and args.mode == 'eval':
        #     rdir = os.path.join(args.sdir, 'views')
        #     print (f'Saving images in {rdir}')
        #     os.makedirs(rdir, exist_ok=True)
        #     pds = render_views(gaussians, cameras, renderArgs)
        #     zero_nums = len(str(len(pds)))
        #     # 将张量转换为图像格式
        #     pairs = zip(np.uint8(pds.permute(0, 2, 3, 1).cpu().numpy() * 255.), np.uint8(gts.permute(0, 2, 3, 1).cpu().numpy() * 255.))  
        #     for idx, (pd, gt) in enumerate(pairs):
        #         Image.fromarray(pd).save(os.path.join(rdir, f'{str(idx).zfill(zero_nums)}_ours.png'))
        #         Image.fromarray(gt).save(os.path.join(rdir, f'{str(idx).zfill(zero_nums)}_gt.png'))
        if args.save_map and args.mode == 'eval':
            rdir = os.path.join(args.sdir, 'views')
            print(f'Saving images in {rdir}')
            os.makedirs(rdir, exist_ok=True)

            pds = render_views(gaussians, cameras, renderArgs)

            zero_nums = len(str(len(pds)))

            # 论文里的差异图逻辑：abs(ours - gt) × 10
            diff_scale = 10.0
            diffs = torch.clamp(torch.abs(pds - gts) * diff_scale, 0.0, 1.0)

            pds_np = np.uint8(torch.clamp(pds, 0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255.0)
            gts_np = np.uint8(torch.clamp(gts, 0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255.0)
            diffs_np = np.uint8(diffs.permute(0, 2, 3, 1).cpu().numpy() * 255.0)

            for idx in range(len(pds_np)):
                name = str(idx).zfill(zero_nums)

                ours_img = Image.fromarray(pds_np[idx])
                gt_img = Image.fromarray(gts_np[idx])
                diff_img = Image.fromarray(diffs_np[idx])

                ours_img.save(os.path.join(rdir, f'{name}_ours.png'))
                gt_img.save(os.path.join(rdir, f'{name}_gt.png'))
                diff_img.save(os.path.join(rdir, f'{name}_diff_x10.png'))

                # 额外保存横向拼接图，方便论文可视化
                w, h = gt_img.size
                pad = 10
                title_h = 30

                panel = Image.new('RGB', (w * 3 + pad * 4, h + title_h + pad * 2), color=(255, 255, 255))

                panel.paste(gt_img, (pad, title_h + pad))
                panel.paste(ours_img, (w + pad * 2, title_h + pad))
                panel.paste(diff_img, (w * 2 + pad * 3, title_h + pad))

                from PIL import ImageDraw
                draw = ImageDraw.Draw(panel)
                draw.text((pad, 8), 'Groundtruth', fill=(0, 0, 0))
                draw.text((w + pad * 2, 8), 'Ours', fill=(0, 0, 0))
                draw.text((w * 2 + pad * 3, 8), '|Ours - GT| x10', fill=(0, 0, 0))

                panel.save(os.path.join(rdir, f'{name}_panel.png'))


def evaluation_o(gaussians, sh_offsets, cameras, attacker, renderArgs, model, message, gts, iteration, eval_interval, args):
    """
    评估模型性能
    
    Args:
        gaussians: 高斯模型对象
        sh_offsets: 球谐函数偏移量
        cameras: 相机列表
        attacker: 攻击器对象
        renderArgs: 渲染参数
        model: 水印解码器模型
        message: 水印消息
        gts: 真实图像
        iteration: 当前迭代次数
        eval_interval: 评估间隔
        args: 参数对象
    """
    # 在指定间隔或评估模式下进行评估
    if iteration % eval_interval == 0 or args.mode == 'eval':
        # 渲染当前模型的图像
        pds = gts #render_views(gaussians, cameras, renderArgs)

        # 计算基本评估指标
        ans = {
            'Bit Acc' : eval_extraction_accuracy(pds, message, model, args),
            **eval_image_quality(pds, gts)
        }

        # 如果指定了攻击类型，计算攻击下的准确率
        if args.atypes is not None:
            ans.update(eval_extraction_accuracy_under_attacks(pds, message, attacker, model, args))
            
        # 格式化并打印日志
        logs = reduce(lambda x1, x2 : f'{x1} | {x2}', [f'{k} : {v:.4f}' for k, v in ans.items()])
        print (f'[ITER {iteration}] {logs}' if args.mode == 'train' else logs)

        #save scores
        if args.mode == 'train':
            with open(os.path.join(args.sdir, 'train_scores.log'), 'a') as txtfile:
                txtfile.write(f'[ITER {iteration}] {logs}\n')
        else:
            with open(os.path.join(args.sdir, 'eval_scores.log'), 'a') as txtfile:
                txtfile.write(f'[ITER {iteration}] {logs}\n')

        torch.cuda.empty_cache()
        
        # 在评估模式下保存图像对比
        if args.save_map and args.mode == 'eval':
            rdir = os.path.join(args.sdir, 'views')
            print (f'Saving images in {rdir}')
            os.makedirs(rdir, exist_ok=True)
            pds = render_views(gaussians, cameras, renderArgs)
            zero_nums = len(str(len(pds)))
            # 将张量转换为图像格式
            pairs = zip(np.uint8(pds.permute(0, 2, 3, 1).cpu().numpy() * 255.), np.uint8(gts.permute(0, 2, 3, 1).cpu().numpy() * 255.))  
            for idx, (pd, gt) in enumerate(pairs):
                Image.fromarray(pd).save(os.path.join(rdir, f'{str(idx).zfill(zero_nums)}_ours.png'))
                Image.fromarray(gt).save(os.path.join(rdir, f'{str(idx).zfill(zero_nums)}_gt.png'))

def evaluation_n_gts(gaussians, sh_offsets, cameras, attacker, renderArgs, model, message, gts, iteration, eval_interval, args):
    """
    评估模型性能
    
    Args:
        gaussians: 高斯模型对象
        sh_offsets: 球谐函数偏移量
        cameras: 相机列表
        attacker: 攻击器对象
        renderArgs: 渲染参数
        model: 水印解码器模型
        message: 水印消息
        gts: 真实图像
        iteration: 当前迭代次数
        eval_interval: 评估间隔
        args: 参数对象
    """
    # 在指定间隔或评估模式下进行评估
    if iteration % eval_interval == 0 or args.mode == 'eval':
        # 渲染当前模型的图像
        gt_noise = RandomGaussianNoise(sigma_min=0.0, sigma_max=0.05)
        pds = gt_noise(gts) #render_views(gaussians, cameras, renderArgs)

        # 计算基本评估指标
        ans = {
            'Bit Acc' : eval_extraction_accuracy(pds, message, model, args),
            **eval_image_quality(pds, gts)
        }

        # 如果指定了攻击类型，计算攻击下的准确率
        if args.atypes is not None:
            ans.update(eval_extraction_accuracy_under_attacks(pds, message, attacker, model, args))
            
        # 格式化并打印日志
        logs = reduce(lambda x1, x2 : f'{x1} | {x2}', [f'{k} : {v:.4f}' for k, v in ans.items()])
        print (f'[ITER {iteration}] {logs}' if args.mode == 'train' else logs)

        #save scores
        if args.mode == 'train':
            with open(os.path.join(args.sdir, 'train_scores.log'), 'a') as txtfile:
                txtfile.write(f'[ITER {iteration}] {logs}\n')
        else:
            with open(os.path.join(args.sdir, 'eval_scores.log'), 'a') as txtfile:
                txtfile.write(f'[ITER {iteration}] {logs}\n')

        torch.cuda.empty_cache()
        
        # 在评估模式下保存图像对比
        if args.save_map and args.mode == 'eval':
            rdir = os.path.join(args.sdir, 'views')
            print (f'Saving images in {rdir}')
            os.makedirs(rdir, exist_ok=True)
            pds = render_views(gaussians, cameras, renderArgs)
            zero_nums = len(str(len(pds)))
            # 将张量转换为图像格式
            pairs = zip(np.uint8(pds.permute(0, 2, 3, 1).cpu().numpy() * 255.), np.uint8(gts.permute(0, 2, 3, 1).cpu().numpy() * 255.))  
            for idx, (pd, gt) in enumerate(pairs):
                Image.fromarray(pd).save(os.path.join(rdir, f'{str(idx).zfill(zero_nums)}_ours.png'))
                Image.fromarray(gt).save(os.path.join(rdir, f'{str(idx).zfill(zero_nums)}_gt.png'))

def evaluation_n_pds(gaussians, sh_offsets, cameras, attacker, renderArgs, model, message, gts, iteration, eval_interval, args):
    """
    评估模型性能
    
    Args:
        gaussians: 高斯模型对象
        sh_offsets: 球谐函数偏移量
        cameras: 相机列表
        attacker: 攻击器对象
        renderArgs: 渲染参数
        model: 水印解码器模型
        message: 水印消息
        gts: 真实图像
        iteration: 当前迭代次数
        eval_interval: 评估间隔
        args: 参数对象
    """
    # 在指定间隔或评估模式下进行评估
    if iteration % eval_interval == 0 or args.mode == 'eval':
        # 渲染当前模型的图像
        pds = render_views(gaussians, cameras, renderArgs)
        gt_noise = RandomGaussianNoise(sigma_min=0.0, sigma_max=0.05)
        pds = gt_noise(pds) #render_views(gaussians, cameras, renderArgs)

        # 计算基本评估指标
        ans = {
            'Bit Acc' : eval_extraction_accuracy(pds, message, model, args),
            **eval_image_quality(pds, gts)
        }

        # 如果指定了攻击类型，计算攻击下的准确率
        if args.atypes is not None:
            ans.update(eval_extraction_accuracy_under_attacks(pds, message, attacker, model, args))
            
        # 格式化并打印日志
        logs = reduce(lambda x1, x2 : f'{x1} | {x2}', [f'{k} : {v:.4f}' for k, v in ans.items()])
        print (f'[ITER {iteration}] {logs}' if args.mode == 'train' else logs)

        #save scores
        if args.mode == 'train':
            with open(os.path.join(args.sdir, 'train_scores.log'), 'a') as txtfile:
                txtfile.write(f'[ITER {iteration}] {logs}\n')
        else:
            with open(os.path.join(args.sdir, 'eval_scores.log'), 'a') as txtfile:
                txtfile.write(f'[ITER {iteration}] {logs}\n')

        torch.cuda.empty_cache()
        
        # 在评估模式下保存图像对比
        if args.save_map and args.mode == 'eval':
            rdir = os.path.join(args.sdir, 'views')
            print (f'Saving images in {rdir}')
            os.makedirs(rdir, exist_ok=True)
            pds = render_views(gaussians, cameras, renderArgs)
            zero_nums = len(str(len(pds)))
            # 将张量转换为图像格式
            pairs = zip(np.uint8(pds.permute(0, 2, 3, 1).cpu().numpy() * 255.), np.uint8(gts.permute(0, 2, 3, 1).cpu().numpy() * 255.))  
            for idx, (pd, gt) in enumerate(pairs):
                Image.fromarray(pd).save(os.path.join(rdir, f'{str(idx).zfill(zero_nums)}_ours.png'))
                Image.fromarray(gt).save(os.path.join(rdir, f'{str(idx).zfill(zero_nums)}_gt.png'))


# 评估视频输出
@torch.no_grad()
def eval(dataset, opt, pipe, hyper, args, model, message):
     # 加载场景与模型
    scene, gaussians, pipe, hyper, info = load_4dgs_scene(args, iteration=args.iteration)
    guardsplat = deepcopy(gaussians)

    # 加载训练时保存的 message
    with open(os.path.join(args.sdir, 'message.txt'), 'r') as txtfile:
        message_text = txtfile.read()
        message = torch.tensor([int(bit) for bit in message_text]).cuda()
        print (f'message : {message_text}')
     
    
    # 4️⃣ 应用 SH 偏移SH_KEYS
    sh_features = {key : getattr(guardsplat, f'_{key}').clone() for key in SH_KEYS}
    
    # ✅ 加载 checkpoint（统一从 checkpoints 文件夹）
    ckpt_dir = os.path.join(args.sdir, 'checkpoints')
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"No checkpoint directory found: {ckpt_dir}")
    pkl_files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith('.pkl'))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl checkpoint found in {ckpt_dir}")
    filename = (
        pkl_files[-1]
        if args.resume_num is None
        else f"sh_offset_{str(args.resume_num).zfill(len(str(args.num_epochs2)))}.pkl"
    )
    ckpt_path = os.path.join(ckpt_dir, filename)
    sh_offsets = torch.load(ckpt_path, weights_only=True, map_location='cuda')

    # 应用 SH 偏移SH_KEYS
    [setattr(guardsplat, f'_{key}', sh_features[key] + sh_offsets[key]) for key in SH_KEYS]

    # print(f"[INFO] Resuming from {ckpt_path}")
    
    # 6️⃣ 设置背景与相机
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
    Cameras = scene.getTestCameras()

    # 7️⃣ 渲染 ground truth
    gts = render_views(gaussians, Cameras, (pipe, background))
    # test_attacker = Attacker(args.atypes, False)
    test_attacker = Attacker(args.atypes, False) if args.atypes is not None else None

    # 执行评估
    evaluation(guardsplat, sh_offsets, Cameras, test_attacker, (pipe, background), model, message, gts, 0, args.eval_interval, args)

    # print(f"[INFO] Evaluation complete using checkpoint: {filename}")     

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4DGS Watermark Trainer")

    # ✅ 初始化 4DGS 参数模块
    mo = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    mh = ModelHiddenParams(parser)

    # 参数
    parser.add_argument('--mode', choices=['train', 'eval'], required=True)
    parser.add_argument('--msg_len', type=int, choices=[16, 32, 48, 64, 72], help='水印消息长度')
    parser.add_argument('--checkpoint_num', type=int, default=20000, help='检查点编号')
    parser.add_argument('--sdir', type=str, default='./Watermark_4D')
    parser.add_argument('--atypes', type=str, nargs='+', default=None)
    parser.add_argument('--learning_rate', type=float, default=5e-3)
    parser.add_argument('--k_rest', type=float, default=20)
    parser.add_argument('--num_epochs2', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=10)
    
    parser.add_argument('--decoder_path', type=str, default="decoders/divid", help='水印解码器路径')
    
    parser.add_argument('--new_lambda_dssim', type=float, default=0.2)
    parser.add_argument('--new_lambda_lpips', type=float, default=1, help='LPIPS 权重')
    parser.add_argument('--lambda_msg', type=float, default=0.1)
    parser.add_argument('--lambda_off', type=float, default=10)
    parser.add_argument('--lambda_recon', type=float, default=1)
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--dataset_type', choices=['blender', 'nerfies', 'dynerf'], default='blender')
    # parser.add_argument('--num_render', type=int, default=10, help='渲染的帧数（默认全部）')
    parser.add_argument('--sample_uniform', action='store_true', help='是否均匀抽样而不是取前几帧')
    parser.add_argument('--train_num_views_per_time', type=int, default=196)
    parser.add_argument('--train_num_frame', type=int, default=25)
    parser.add_argument('--train_random_sample_view', type=bool, default=False)
    parser.add_argument('--train_random_sample_time', type=bool, default=False)
    
    parser.add_argument('--configs', type=str, default=None)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--iteration', type=int, default=-1)
    parser.add_argument('--train_num_views_per_time_real', type=int, default=100)
    parser.add_argument('--train_num_frame_real', type=int, default=8)
    parser.add_argument('--train_random_sample_view_real', type=bool, default=True)
    parser.add_argument('--train_random_sample_time_real', type=bool, default=True)
    parser.add_argument('--train_random_sample_time_real_interval', type=bool, default=True)
    
    parser.add_argument('--test_num_views_per_time', type=int, default=100)
    parser.add_argument('--test_num_frame', type=int, default=8)
    parser.add_argument('--test_random_sample_view', type=bool, default=False)
    parser.add_argument('--test_random_sample_time', type=bool, default=False)

    # 评估相关参数
    parser.add_argument('--resume_num', type=int, default=None, help='恢复的检查点编号')
    parser.add_argument('--save_map', action='store_true', help='保存图像对比')
    
    
    # ✅ 提取合并参数
    args = get_combined_args(parser)
    
    import mmengine

    if args.configs:
        cfg = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, cfg)
        # breakpoint()
        # print("[DEBUG] dataset_type after merge:", args.dataset_type)
        # print("[DEBUG] source_path:", args.source_path)

        
        # ✅ 覆盖 dataset_type，保持与训练一致
        if hasattr(cfg, 'dataset_type'):
            args.dataset_type = cfg.dataset_type
            # print(f"[INFO] Using dataset_type from configs: {args.dataset_type}")
    else:
        cfg = None
        


    # ✅ ---- 修补逻辑：防止旧 cfg_args 丢失新参数 ----
    if not hasattr(args, "atypes"):
        args.atypes = None
    if not hasattr(args, "resume_num"):
        args.resume_num = None
    if not hasattr(args, "iteration"):
        args.iteration = -1

    # 根据模式打印相应信息
    if args.mode == 'train':
        pstr = 'Watermarking'
        print (f'Watermarking {args.model_path} with N_L={args.msg_len}')
        print (f'GuardSplat will be saved in {args.sdir}')
    else:
        pstr = 'Evaluation'
        print (f'Evaluating GuardSplat in {args.sdir}')

    # ✅ 标准预处理
    args.model_path = os.path.abspath(args.model_path)
    args.source_path = os.path.abspath(args.source_path)
    wm_params = load_decoder_and_message(args.msg_len, args.decoder_path)
    safe_state(args.quiet)

    #########################################################################################
    #init seed
    # seed=42
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark=False

    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    ###############################################################################




    print(f"[INFO] >>> {'Training' if args.mode=='train' else 'Evaluating'} Watermarked 4DGS <<<")

    # ✅ 调用对应模式
    globals()[args.mode](
        mo.extract(args),
        op.extract(args),
        pp.extract(args),
        mh.extract(args),
        args,
        *wm_params
    )
