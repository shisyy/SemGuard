import os
import torch
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '4DGaussians'))
from arguments import ModelParams, PipelineParams, ModelHiddenParams
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state

import argparse
# 核心函数
def load_4dgs_scene(args, iteration=-1):
    """
    加载已训练好的 4D Gaussian Splatting 场景和模型。

    Args:
        args: argparse.Namespace, 包含 model_path, configs 等
        iteration: int, 载入的 checkpoint 迭代次数 (-1 表示自动)

    Returns:
        scene: Scene 对象
        gaussians: GaussianModel 对象
        info: dict, 场景基础信息
    """
    # ✅ 创建 parser 仅用于注册参数组（4DGS 内部要求）
    parser = argparse.ArgumentParser()
    model = ModelParams(parser)
    pipe = PipelineParams(parser)
    hyper = ModelHiddenParams(parser)

    # ✅ 从 args 提取参数
    model = model.extract(args)
    pipe = pipe.extract(args)
    hyper = hyper.extract(args)

    print(f"[INFO] Loading 4DGS scene from: {model.model_path}")

    # ✅ 加载 Gaussian 模型和 Scene
    gaussians = GaussianModel(model.sh_degree, args)
    scene = Scene(model, gaussians, load_iteration=iteration, shuffle=False, gen=True, gen_dict = {
        # "dataset_type": getattr(args, "dataset_type", None),
        "train_num_views_per_time":getattr(args, "train_num_views_per_time", 196),
        "train_num_frame":getattr(args, "train_num_frame", 25),
        "train_random_sample_view":getattr(args, "train_random_sample_view", True),
        "train_random_sample_time":getattr(args, "train_random_sample_time", False),
        "test_num_views_per_time":getattr(args, "test_num_views_per_time", 100),
        "test_num_frame":getattr(args, "test_num_frame", 8),
        "test_random_sample_view":getattr(args, "test_random_sample_view", False),
        "test_random_sample_time":getattr(args, "test_random_sample_time", False)
    })

    

    cam_type = scene.dataset_type
    print(f"[INFO] Dataset type: {cam_type}")

    info = {
        "model_path": model.model_path,
        "dataset_type": cam_type,
        "extent": scene.cameras_extent,
        "maxtime": scene.maxtime,
        "num_train": len(scene.getTrainCameras()),
        "num_test": len(scene.getTestCameras()),
        "num_video": len(scene.getVideoCameras()),
    }

    print(f"[INFO] Cameras loaded: Train={info['num_train']} | Test={info['num_test']} | Video={info['num_video']}")
    return scene, gaussians, pipe, hyper, info

# 相机访问接口
def get_cameras(scene, mode="video", num_limit=None):
    """
    获取指定类型的相机视角。

    Args:
        scene: 由 load_4dgs_scene() 返回的 Scene 对象
        mode: 'train' | 'test' | 'video'
        num_limit: 限制返回相机数量（None 表示全部）

    Returns:
        List[MiniCam]
    """
    if mode == "train":
        cams = scene.getTrainCameras()
    elif mode == "test":
        cams = scene.getTestCameras()
    elif mode == "video":
        cams = scene.getVideoCameras()
    else:
        raise ValueError(f"Unknown camera mode: {mode}")

    if num_limit is not None:
        cams = cams[:num_limit]
    print(f"[INFO] Loaded {len(cams)} {mode} cameras.")
    return cams

def get_valid_cameras(scene, cam_type="video"):
    """
    获取可用于渲染的相机。
    自动从 FourDGSdataset 生成 Camera 对象。
    """
    video_cams = get_cameras(scene, cam_type)

    # ✅ 如果是 FourDGSdataset，则通过索引访问，触发 __getitem__() → 返回 Camera
    if hasattr(video_cams, "__getitem__") and hasattr(video_cams, "__len__"):
        video_cams = [video_cams[i] for i in range(len(video_cams))]
    elif hasattr(video_cams, "dataset"):
        # 兜底（极少数情况）
        cams_data = video_cams.dataset
        video_cams = [video_cams[i] for i in range(len(cams_data))]
    else:
        raise RuntimeError(f"[FATAL] Unknown camera structure: {type(video_cams)}")

    print(f"[INFO] Loaded {len(video_cams)} cameras for dataset '{scene.dataset_type}'")

    if len(video_cams) == 0:
        raise RuntimeError(f"[FATAL] No cameras found for dataset {scene.dataset_type}")

    return video_cams

# 场景/相机信息打印

def print_scene_info(info):
    """打印场景基本信息"""
    print("\n[INFO] Scene Summary:")
    for k, v in info.items():
        print(f"  {k:12s}: {v}")
    print("---------------------------------------------------------")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="4DGS Dynamic NeRF Utils")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--configs", type=str)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    safe_state(args.quiet)

    # 加载场景与相机
    scene, gaussians, pipe, info = load_4dgs_scene(args, iteration=args.iteration)

    # 打印信息
    print_scene_info(info)
