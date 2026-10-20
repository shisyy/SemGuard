import os
import sys
from PIL import Image
from scene.cameras import Camera

from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.hyper_loader import Load_hyper_data, format_hyper_data
import torchvision.transforms as transforms
import copy
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import torch
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.general_utils import PILtoTorch
from tqdm import tqdm
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    mask: np.array
   
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int



def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    # breakpoint()
    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        image = PILtoTorch(image,None)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              time = float(idx/len(cam_extrinsics)), mask=None) # default by monocular settings.
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    # breakpoint()
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    # breakpoint()
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    
    try:
        pcd = fetchPly(ply_path)
        
    except:
        pcd = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=train_cam_infos,
                           maxtime=0,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info



def generateCamerasFromTransforms_gen_new(
    path,
    template_transformsfile,
    extension,
    maxtime,
    num_views_per_time=100,   # M：指定视角数
    num_frame=8,              # N：指定帧数
    random_sample_view=True,
    random_sample_time=False
):
    print("random_sample_view:", random_sample_view)
    print("random_sample_time:", random_sample_time)
    print("num_views_per_time:", num_views_per_time)
    print("num_frame:", num_frame)
    trans_t = lambda t: torch.Tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]
    ]).float()

    rot_phi = lambda phi: torch.Tensor([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi),  np.cos(phi), 0],
        [0, 0, 0, 1]
    ]).float()

    rot_theta = lambda th: torch.Tensor([
        [ np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [ np.sin(th), 0,  np.cos(th), 0],
        [0, 0, 0, 1]
    ]).float()

    def pose_spherical(theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi / 180. * np.pi) @ c2w
        c2w = rot_theta(theta / 180. * np.pi) @ c2w
        c2w = torch.Tensor(np.array([
            [-1, 0, 0, 0],
            [ 0, 0, 1, 0],
            [ 0, 1, 0, 0],
            [ 0, 0, 0, 1]
        ])).float() @ c2w
        return c2w

    # ---- read template ----
    with open(os.path.join(path, template_transformsfile), "r") as json_file:
        template_json = json.load(json_file)
        try:
            fovx = template_json["camera_angle_x"]
        except Exception:
            fovx = focal2fov(template_json["fl_x"], template_json["w"])

    # ---- load one image ----
    image = None
    for frame in template_json["frames"]:
        cam_name = os.path.join(path, frame["file_path"] + extension)
        image_path = os.path.join(path, cam_name)
        _ = Path(cam_name).stem
        pil_img = Image.open(image_path)
        image = PILtoTorch(pil_img, (800, 800))
        break
    if image is None:
        raise RuntimeError("template_transformsfile 中 frames 为空，无法读取示例图像。")

    M = int(num_views_per_time)  # 视角数
    N = int(num_frame)           # 帧数

    # ✅ 视角角度：random_sample_view=False 时均匀采样（不包含终点避免重复）
    if random_sample_view:
        angles = np.random.uniform(-180.0, 180.0, M)
    else:
        angles = np.linspace(-180.0, 180.0, M + 1)[:-1]  # 均匀 M 个，去掉 180 端点

    # 时间：样本内均匀 N 帧
    if random_sample_time:
        times = np.random.uniform(0, maxtime, N)
        render_times = torch.Tensor(sorted(times)) / maxtime
    else:
        render_times = torch.linspace(0, maxtime, N) / maxtime

    cam_infos_videos = []  # [ [CameraInfo]*N ]*M
    uid = 0

    for view_idx in range(M):
        angle = float(angles[view_idx])
        pose = pose_spherical(angle, -30.0, 4.0)  # 该样本内所有帧同一个视角

        one_video_infos = []
        for frame_idx in range(N):
            time = float(render_times[frame_idx])

            matrix = np.linalg.inv(np.array(pose))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
            FovY = fovy
            FovX = fovx

            one_video_infos.append(
                CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=None,
                    image_name=None,
                    width=image.shape[1],
                    height=image.shape[2],
                    time=time,
                    mask=None
                )
            )
            uid += 1

        cam_infos_videos.append(one_video_infos)

    return cam_infos_videos



def readNerfSyntheticInfo_gen_new(path, white_background, eval, extension=".png", 
        train_num_views_per_time=100, train_num_frame = 160, train_random_sample_view=True, train_random_sample_time=True,
        test_num_views_per_time=100, test_num_frame = 20, test_random_sample_view=False, test_random_sample_time=False):
    timestamp_mapper, max_time = read_timeline(path)
    print("Reading Training Transforms")
    train_cam_infos = generateCamerasFromTransforms_gen_new(path, "transforms_train.json", extension, max_time, num_views_per_time=train_num_views_per_time, num_frame=train_num_frame, random_sample_view=train_random_sample_view, random_sample_time=train_random_sample_time)
    print("Reading Test Transforms")
    test_cam_infos = generateCamerasFromTransforms_gen_new(path, "transforms_train.json", extension, max_time, num_views_per_time=test_num_views_per_time, num_frame=test_num_frame, random_sample_view=test_random_sample_view, random_sample_time=test_random_sample_time)
    print("Generating Video Transforms")
    video_cam_infos = generateCamerasFromTransforms_gen_new(path, "transforms_train.json", extension, max_time)
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []
    if isinstance(train_cam_infos, list):
        nerf_normalization = []
        for cam_info in train_cam_infos:
            nerf_normalization.append(getNerfppNorm(cam_info))
    else:
        nerf_normalization = getNerfppNorm(train_cam_infos)


    ply_path = os.path.join(path, "fused.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 2000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        pcd = fetchPly(ply_path)
        # xyz = -np.array(pcd.points)
        # pcd = pcd._replace(points=xyz)


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time
                           )
    return scene_info



# def generateCamerasFromTransforms_new(path, template_transformsfile, extension, maxtime, num_views=160, random_sample=False):
#     trans_t = lambda t : torch.Tensor([
#     [1,0,0,0],
#     [0,1,0,0],
#     [0,0,1,t],
#     [0,0,0,1]]).float()

#     rot_phi = lambda phi : torch.Tensor([
#         [1,0,0,0],
#         [0,np.cos(phi),-np.sin(phi),0],
#         [0,np.sin(phi), np.cos(phi),0],
#         [0,0,0,1]]).float()

#     rot_theta = lambda th : torch.Tensor([
#         [np.cos(th),0,-np.sin(th),0],
#         [0,1,0,0],
#         [np.sin(th),0, np.cos(th),0],
#         [0,0,0,1]]).float()
#     def pose_spherical(theta, phi, radius):
#         c2w = trans_t(radius)
#         c2w = rot_phi(phi/180.*np.pi) @ c2w
#         c2w = rot_theta(theta/180.*np.pi) @ c2w
#         c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
#         return c2w
#     cam_infos = []
#     # generate render poses and times
#     # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)
    
#     if random_sample:
#         # 随机角度（不重复）
#         angles = np.random.uniform(-180, 180, num_views)
#     else:
#         # 均匀分布角度
#         angles = np.linspace(-180, 180, num_views + 1)[:-1]
#     render_poses = torch.stack(
#         [pose_spherical(angle, -30.0, 4.0) for angle in angles],
#         0
#     )
#     print("max time:", maxtime)
#     render_times = torch.linspace(0,maxtime,render_poses.shape[0])
#     with open(os.path.join(path, template_transformsfile)) as json_file:
#         template_json = json.load(json_file)
#         try:
#             fovx = template_json["camera_angle_x"]
#         except:
#             fovx = focal2fov(template_json["fl_x"], template_json['w'])

#     # load a single image to get image info.
#     for idx, frame in enumerate(template_json["frames"]):
#         cam_name = os.path.join(path, frame["file_path"] + extension)
#         image_path = os.path.join(path, cam_name)
#         image_name = Path(cam_name).stem
#         image = Image.open(image_path)
#         im_data = np.array(image.convert("RGBA"))
#         image = PILtoTorch(image,(800,800))
#         break
#     # format information
#     for idx, (time, poses) in enumerate(zip(render_times,render_poses)):
#         time = time/maxtime
#         matrix = np.linalg.inv(np.array(poses))
#         R = -np.transpose(matrix[:3,:3])
#         R[:,0] = -R[:,0]
#         T = -matrix[:3, 3]
#         fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
#         FovY = fovy 
#         FovX = fovx
#         cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
#                             image_path=None, image_name=None, width=image.shape[1], height=image.shape[2],
#                             time = time, mask=None))
#     return cam_infos

def generateCamerasFromTransforms(path, template_transformsfile, extension, maxtime):
    trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

    rot_phi = lambda phi : torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).float()

    rot_theta = lambda th : torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).float()
    def pose_spherical(theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        return c2w
    cam_infos = []
    # generate render poses and times
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)
    render_times = torch.linspace(0,maxtime,render_poses.shape[0])
    with open(os.path.join(path, template_transformsfile)) as json_file:
        template_json = json.load(json_file)
        try:
            fovx = template_json["camera_angle_x"]
        except:
            fovx = focal2fov(template_json["fl_x"], template_json['w'])

    # load a single image to get image info.
    for idx, frame in enumerate(template_json["frames"]):
        cam_name = os.path.join(path, frame["file_path"] + extension)
        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        image = PILtoTorch(image,(800,800))
        break
    # format information
    for idx, (time, poses) in enumerate(zip(render_times,render_poses)):
        time = time/maxtime
        matrix = np.linalg.inv(np.array(poses))
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
        FovY = fovy 
        FovX = fovx
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=None, image_name=None, width=image.shape[1], height=image.shape[2],
                            time = time, mask=None))
    return cam_infos
def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", mapper = {}):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = focal2fov(contents['fl_x'],contents['w'])
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            time = mapper[frame["time"]]
            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            image = PILtoTorch(image,(800,800))
            fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[1], height=image.shape[2],
                            time = time, mask=None))
            
    return cam_infos
def read_timeline(path):
    with open(os.path.join(path, "transforms_train.json")) as json_file:
        train_json = json.load(json_file)
    with open(os.path.join(path, "transforms_test.json")) as json_file:
        test_json = json.load(json_file)  
    time_line = [frame["time"] for frame in train_json["frames"]] + [frame["time"] for frame in test_json["frames"]]
    time_line = set(time_line)
    time_line = list(time_line)
    time_line.sort()
    timestamp_mapper = {}
    max_time_float = max(time_line)
    for index, time in enumerate(time_line):
        timestamp_mapper[time] = time/max_time_float

    return timestamp_mapper, max_time_float


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    timestamp_mapper, max_time = read_timeline(path)
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, timestamp_mapper)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, timestamp_mapper)
    print("Generating Video Transforms")
    video_cam_infos = generateCamerasFromTransforms(path, "transforms_train.json", extension, max_time)
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "fused.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 2000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        pcd = fetchPly(ply_path)
        # xyz = -np.array(pcd.points)
        # pcd = pcd._replace(points=xyz)


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time
                           )
    return scene_info

def format_infos(dataset,split):
    # loading
    cameras = []
    image = dataset[0][0]
    if split == "train":
        for idx in tqdm(range(len(dataset))):
            image_path = None
            image_name = f"{idx}"
            time = dataset.image_times[idx]
            # matrix = np.linalg.inv(np.array(pose))
            R,T = dataset.load_pose(idx)
            FovX = focal2fov(dataset.focal[0], image.shape[1])
            FovY = focal2fov(dataset.focal[0], image.shape[2])
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))

    return cameras


import numpy as np

def generateCamerasFromHyperInterpolation_gen_new(
    dataset,                 # Load_hyper_data(...) 的实例（train/test都行）
    maxtime,                 # dataset.max_time（通常是 1.0）
    num_views_per_time=100,  # M：视角数（选多少个 camera_id）
    num_frame=8,             # N：每个视角生成多少帧（时间采样）
    random_sample_view=True,
    random_sample_time=False
):
    """
    返回 cam_infos_videos: [ [CameraInfo]*N ]*M
    - “视角”定义为 Hyper 数据里的 camera_id
    - 每个 camera_id 内按时间插值位姿，得到任意时间的连续位姿
    - time 输出为 [0,1]（与你 transforms_gen_new 一致）
    - image 用 dataset.image_one_torch 占位（不读真实图）
    """

    # ----------------- quaternion / slerp -----------------
    def rotmat_to_quat(R):
        m = R
        t = np.trace(m)
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        else:
            i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
            if i == 0:
                s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
                w = (m[2, 1] - m[1, 2]) / s
                x = 0.25 * s
                y = (m[0, 1] + m[1, 0]) / s
                z = (m[0, 2] + m[2, 0]) / s
            elif i == 1:
                s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
                w = (m[0, 2] - m[2, 0]) / s
                x = (m[0, 1] + m[1, 0]) / s
                y = 0.25 * s
                z = (m[1, 2] + m[2, 1]) / s
            else:
                s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
                w = (m[1, 0] - m[0, 1]) / s
                x = (m[0, 2] + m[2, 0]) / s
                y = (m[1, 2] + m[2, 1]) / s
                z = 0.25 * s
        q = np.array([w, x, y, z], dtype=np.float64)
        return q / (np.linalg.norm(q) + 1e-12)

    def quat_to_rotmat(q):
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ], dtype=np.float64)

    def slerp(q0, q1, t):
        q0 = q0 / (np.linalg.norm(q0) + 1e-12)
        q1 = q1 / (np.linalg.norm(q1) + 1e-12)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        dot = np.clip(dot, -1.0, 1.0)
        if dot > 0.9995:
            q = q0 + t * (q1 - q0)
            return q / (np.linalg.norm(q) + 1e-12)
        theta0 = np.arccos(dot)
        sin0 = np.sin(theta0)
        theta = theta0 * t
        s0 = np.sin(theta0 - theta) / (sin0 + 1e-12)
        s1 = np.sin(theta) / (sin0 + 1e-12)
        return s0 * q0 + s1 * q1

    # ----------------- SE(3) helpers -----------------
    def make_4x4(R, t):
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = R
        M[:3, 3] = t
        return M

    def invert_4x4(M):
        R = M[:3, :3]
        t = M[:3, 3]
        Mi = np.eye(4, dtype=np.float64)
        Mi[:3, :3] = R.T
        Mi[:3, 3] = -R.T @ t
        return Mi

    def interpolate_c2w(c2w0, c2w1, alpha):
        R0, t0 = c2w0[:3, :3], c2w0[:3, 3]
        R1, t1 = c2w1[:3, :3], c2w1[:3, 3]
        q0 = rotmat_to_quat(R0)
        q1 = rotmat_to_quat(R1)
        q = slerp(q0, q1, alpha)
        R = quat_to_rotmat(q)
        t = (1.0 - alpha) * t0 + alpha * t1
        return make_4x4(R, t)

    # ----------------- Hyper camera -> (R,T)（沿用你现有读取逻辑） -----------------
    def hyper_camera_to_RT(hyper_cam):
        R = np.array(hyper_cam.orientation.T, dtype=np.float64)
        T = np.array(- hyper_cam.position @ R, dtype=np.float64)
        return R, T

    def find_neighbors(sorted_times, t):
        if t <= sorted_times[0]:
            return 0, 0, 0.0
        if t >= sorted_times[-1]:
            n = len(sorted_times) - 1
            return n, n, 0.0
        j = int(np.searchsorted(sorted_times, t, side="right"))
        i0, i1 = j - 1, j
        t0, t1 = float(sorted_times[i0]), float(sorted_times[i1])
        a = (t - t0) / (t1 - t0 + 1e-12)
        return i0, i1, float(a)

    # ----------------- choose views (camera_id) -----------------
    all_cam_ids = np.array(dataset.all_cam, dtype=np.int64)
    all_times = np.array(dataset.all_time, dtype=np.float64)  # 已归一化 [0,1]
    unique_cam_ids = np.unique(all_cam_ids)

    # 只保留拥有>=2帧的 camera_id（否则无法插值）
    valid_cam_ids = []
    camid_to_indices_sorted = {}
    camid_to_times_sorted = {}

    for cid in unique_cam_ids:
        idxs = np.where(all_cam_ids == cid)[0]
        if len(idxs) < 2:
            continue
        ts = all_times[idxs]
        order = np.argsort(ts)
        idxs_sorted = idxs[order]
        ts_sorted = ts[order]
        cid = int(cid)
        valid_cam_ids.append(cid)
        camid_to_indices_sorted[cid] = idxs_sorted
        camid_to_times_sorted[cid] = ts_sorted

    if len(valid_cam_ids) == 0:
        raise RuntimeError("没有任何 camera_id 拥有>=2帧，无法做插值。")

    M = min(int(num_views_per_time), len(valid_cam_ids))
    if random_sample_view:
        chosen_cam_ids = list(np.random.choice(valid_cam_ids, size=M, replace=False))
    else:
        # 均匀选取
        pick = np.linspace(0, len(valid_cam_ids) - 1, M).round().astype(int)
        chosen_cam_ids = [valid_cam_ids[i] for i in pick]

    # ----------------- time samples -----------------
    N = int(num_frame)
    if random_sample_time:
        render_times = np.sort(np.random.uniform(0.0, 1.0, N)).astype(np.float64)
    else:
        render_times = np.linspace(0.0, 1.0, N).astype(np.float64)

    # ----------------- shared intrinsics / placeholder image -----------------
    # 不读取真实图像，用占位图即可
    image = dataset.image_one_torch
    h, w = dataset.h, dataset.w
    template_cam = dataset.all_cam_params[0]
    FovY = focal2fov(template_cam.focal_length, h)
    FovX = focal2fov(template_cam.focal_length, w)

    # ----------------- build output -----------------
    cam_infos_videos = []  # [ [CameraInfo]*N ]*M
    uid = 0

    for view_idx in range(M):
        cid = chosen_cam_ids[view_idx]
        idxs_sorted = camid_to_indices_sorted[cid]
        times_sorted = camid_to_times_sorted[cid]

        one_video_infos = []
        for frame_idx in range(N):
            time = float(render_times[frame_idx])  # ✅ 与 transforms_gen_new 一致：归一化 [0,1]

            i0, i1, a = find_neighbors(times_sorted, time)
            idx0 = int(idxs_sorted[i0])
            idx1 = int(idxs_sorted[i1])

            cam0 = dataset.all_cam_params[idx0]
            cam1 = dataset.all_cam_params[idx1]
            R0, T0 = hyper_camera_to_RT(cam0)
            R1, T1 = hyper_camera_to_RT(cam1)

            # 认为 (R,T) 是 w2c：x_cam = R x_world + T
            w2c0 = make_4x4(R0, T0)
            w2c1 = make_4x4(R1, T1)
            c2w0 = invert_4x4(w2c0)
            c2w1 = invert_4x4(w2c1)

            # 在 c2w 上插值，再转回 w2c 输出 R,T
            c2w = interpolate_c2w(c2w0, c2w1, a)
            w2c = invert_4x4(c2w)

            R = w2c[:3, :3]
            T = w2c[:3, 3]

            one_video_infos.append(
                CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,        # ✅ 占位
                    image_path=None,
                    image_name=None,
                    width=w,
                    height=h,
                    time=time * float(maxtime),  # maxtime 通常=1；保持接口一致
                    mask=None
                )
            )
            uid += 1

        cam_infos_videos.append(one_video_infos)

    return cam_infos_videos

# def generateCamerasFromHyper_gen_new(
#     dataset,                 # 必须是 Load_hyper_data 的实例
#     maxtime,                 # dataset.max_time（一般=1）
#     num_views_per_time=100,  # M
#     num_frame=8,             # N
#     random_sample_view=True,
#     random_sample_time=False
# ):
#     """
#     输出 cam_infos_videos: [ [CameraInfo]*N ]*M
#     - view 维度：按 camera_id 选 M 个“真实视角”
#     - time 维度：采样 N 个时间点（均匀/随机），在每个 view 的时间序列里插值位姿
#     - 不读真实图像：CameraInfo.image 用 dataset.image_one_torch 占位
#     """

#     # --- 强校验：避免你再传错 list ---
#     if not hasattr(dataset, "all_cam") or not hasattr(dataset, "all_time") or not hasattr(dataset, "all_cam_params"):
#         raise TypeError(
#             f"dataset 必须是 Load_hyper_data(...) 的实例，当前类型：{type(dataset)}。"
#             "不要传 format_hyper_data(...) 的返回(list)。"
#         )

#     # ===== 1) 按 camera_id 分组，并按时间排序 =====
#     all_cam_ids = np.array(dataset.all_cam, dtype=np.int64)
#     all_times = np.array(dataset.all_time, dtype=np.float64)  # [0,1]
#     unique_cam_ids = np.unique(all_cam_ids)

#     camid_to_sorted_indices = {}
#     camid_to_sorted_times = {}
#     valid_cam_ids = []

#     for cid in unique_cam_ids:
#         idxs = np.where(all_cam_ids == cid)[0]
#         if len(idxs) < 2:
#             continue
#         ts = all_times[idxs]
#         order = np.argsort(ts)
#         idxs_sorted = idxs[order]
#         ts_sorted = ts[order]
#         cid = int(cid)
#         valid_cam_ids.append(cid)
#         camid_to_sorted_indices[cid] = idxs_sorted
#         camid_to_sorted_times[cid] = ts_sorted

#     if len(valid_cam_ids) == 0:
#         raise RuntimeError("没有任何 camera_id 拥有>=2帧，无法插值生成。")

#     # ===== 2) 选 M 个视角（camera_id）=====
#     M = min(int(num_views_per_time), len(valid_cam_ids))
#     if random_sample_view:
#         chosen_cam_ids = list(np.random.choice(valid_cam_ids, size=M, replace=False))
#     else:
#         pick = np.linspace(0, len(valid_cam_ids) - 1, M).round().astype(int)
#         chosen_cam_ids = [valid_cam_ids[i] for i in pick]

#     # ===== 3) 生成 N 个时间采样点 =====
#     N = int(num_frame)
#     if random_sample_time:
#         render_times = np.sort(np.random.uniform(0.0, 1.0, N)).astype(np.float64)
#     else:
#         render_times = np.linspace(0.0, 1.0, N).astype(np.float64)

#     # ===== 4) 准备：位姿插值（四元数 slerp + 平移线性）=====
#     def rotmat_to_quat(R):
#         m = R
#         t = np.trace(m)
#         if t > 0:
#             s = np.sqrt(t + 1.0) * 2
#             w = 0.25 * s
#             x = (m[2, 1] - m[1, 2]) / s
#             y = (m[0, 2] - m[2, 0]) / s
#             z = (m[1, 0] - m[0, 1]) / s
#         else:
#             i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
#             if i == 0:
#                 s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
#                 w = (m[2, 1] - m[1, 2]) / s
#                 x = 0.25 * s
#                 y = (m[0, 1] + m[1, 0]) / s
#                 z = (m[0, 2] + m[2, 0]) / s
#             elif i == 1:
#                 s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
#                 w = (m[0, 2] - m[2, 0]) / s
#                 x = (m[0, 1] + m[1, 0]) / s
#                 y = 0.25 * s
#                 z = (m[1, 2] + m[2, 1]) / s
#             else:
#                 s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
#                 w = (m[1, 0] - m[0, 1]) / s
#                 x = (m[0, 2] + m[2, 0]) / s
#                 y = (m[1, 2] + m[2, 1]) / s
#                 z = 0.25 * s
#         q = np.array([w, x, y, z], dtype=np.float64)
#         return q / (np.linalg.norm(q) + 1e-12)

#     def quat_to_rotmat(q):
#         w, x, y, z = q
#         return np.array([
#             [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
#             [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
#             [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
#         ], dtype=np.float64)

#     def slerp(q0, q1, t):
#         q0 = q0 / (np.linalg.norm(q0) + 1e-12)
#         q1 = q1 / (np.linalg.norm(q1) + 1e-12)
#         dot = float(np.dot(q0, q1))
#         if dot < 0.0:
#             q1 = -q1
#             dot = -dot
#         dot = np.clip(dot, -1.0, 1.0)
#         if dot > 0.9995:
#             q = q0 + t * (q1 - q0)
#             return q / (np.linalg.norm(q) + 1e-12)
#         theta0 = np.arccos(dot)
#         sin0 = np.sin(theta0)
#         theta = theta0 * t
#         s0 = np.sin(theta0 - theta) / (sin0 + 1e-12)
#         s1 = np.sin(theta) / (sin0 + 1e-12)
#         return s0 * q0 + s1 * q1

#     def interpolate_pose(R0, T0, R1, T1, a):
#         q0 = rotmat_to_quat(R0)
#         q1 = rotmat_to_quat(R1)
#         q = slerp(q0, q1, a)
#         R = quat_to_rotmat(q)
#         T = (1.0 - a) * T0 + a * T1
#         return R, T

#     def find_neighbors(ts_sorted, t):
#         if t <= ts_sorted[0]:
#             return 0, 0, 0.0
#         if t >= ts_sorted[-1]:
#             n = len(ts_sorted) - 1
#             return n, n, 0.0
#         j = int(np.searchsorted(ts_sorted, t, side="right"))
#         i0, i1 = j - 1, j
#         t0, t1 = float(ts_sorted[i0]), float(ts_sorted[i1])
#         a = (t - t0) / (t1 - t0 + 1e-12)
#         return i0, i1, float(a)

#     # ===== 5) 生成 cam_infos_videos =====
#     cam_infos_videos = []
#     uid = 0

#     for view_idx in range(M):
#         cid = chosen_cam_ids[view_idx]
#         idxs_sorted = camid_to_sorted_indices[cid]
#         ts_sorted = camid_to_sorted_times[cid]

#         one_video_infos = []
#         for frame_idx in range(N):
#             t = float(render_times[frame_idx])  # 归一化时间 [0,1]

#             i0, i1, a = find_neighbors(ts_sorted, t)
#             idx0 = int(idxs_sorted[i0])
#             idx1 = int(idxs_sorted[i1])

#             # 取两帧的 R,T（沿用 dataset 的定义）
#             cam0 = dataset.all_cam_params[idx0]
#             cam1 = dataset.all_cam_params[idx1]
#             R0 = cam0.orientation.T
#             T0 = - cam0.position @ R0
#             R1 = cam1.orientation.T
#             T1 = - cam1.position @ R1

#             R, T = interpolate_pose(
#                 np.array(R0, np.float64), np.array(T0, np.float64),
#                 np.array(R1, np.float64), np.array(T1, np.float64),
#                 a
#             )

#             # 用 dataset 内提供的方法构造 CameraInfo（只需位姿+时间）
#             one_video_infos.append(
#                 dataset.load_pose_time_only(
#                     idx0,                    # 用 idx0 作为模板取 FOV/尺寸等
#                     uid=uid,
#                     time_override=t * float(maxtime)
#                 )._replace(R=R, T=T)  # 如果 CameraInfo 是 namedtuple，可用 _replace；如果是 class，见下方说明
#             )
#             uid += 1

#         cam_infos_videos.append(one_video_infos)
#     breakpoint()
#     return cam_infos_videos

import numpy as np

def generateCamerasFromHyper_gen_new(
    dataset,                 # 必须是 Load_hyper_data 的实例
    maxtime,                 # dataset.max_time（一般=1）
    num_views_per_time=100,  # M
    num_frame=8,             # N
    random_sample_view=True,
    random_sample_time=False
):
    """
    输出 cam_infos_videos: [ [CameraInfo]*N ]*M

    - view 维度：优先选真实 camera_id；如果真实视角不足，则在真实相机之间插值得到更多“虚拟视角”
    - time 维度：采样 N 个时间点（均匀/随机），在每个 view 的时间序列里插值位姿
    - 不读真实图像：CameraInfo.image 用 dataset.image_one_torch 占位（由 dataset.load_pose_time_only 提供）

    说明：
    - 虚拟视角插值：在同一时刻 t，先得到两个真实相机的 c2w（orientation/position）并在 view 维度做 slerp+线性插值；
      然后转换为 w2c 的 (R, T) 写回 CameraInfo。
    - 若 CameraInfo 不是 namedtuple，没有 _replace，则会 fallback 到直接 setattr。
    """

    # --- 强校验：避免你再传错 list ---
    if not hasattr(dataset, "all_cam") or not hasattr(dataset, "all_time") or not hasattr(dataset, "all_cam_params"):
        raise TypeError(
            f"dataset 必须是 Load_hyper_data(...) 的实例，当前类型：{type(dataset)}。"
            "不要传 format_hyper_data(...) 的返回(list)。"
        )

    # ===== 1) 按 camera_id 分组，并按时间排序 =====
    all_cam_ids = np.array(dataset.all_cam, dtype=np.int64)
    all_times = np.array(dataset.all_time, dtype=np.float64)  # 归一化时间 [0,1]
    unique_cam_ids = np.unique(all_cam_ids)

    camid_to_sorted_indices = {}
    camid_to_sorted_times = {}
    valid_cam_ids = []

    for cid in unique_cam_ids:
        idxs = np.where(all_cam_ids == cid)[0]
        if len(idxs) < 2:
            continue
        ts = all_times[idxs]
        order = np.argsort(ts)
        idxs_sorted = idxs[order]
        ts_sorted = ts[order]
        cid = int(cid)
        valid_cam_ids.append(cid)
        camid_to_sorted_indices[cid] = idxs_sorted
        camid_to_sorted_times[cid] = ts_sorted

    if len(valid_cam_ids) == 0:
        raise RuntimeError("没有任何 camera_id 拥有>=2帧，无法插值生成。")

    # ===== 2) 生成 N 个时间采样点 =====
    N = int(num_frame)
    if random_sample_time:
        render_times = np.sort(np.random.uniform(0.0, 1.0, N)).astype(np.float64)
    else:
        render_times = np.linspace(0.0, 1.0, N).astype(np.float64)

    # ===== 3) 工具：旋转/四元数/插值 =====
    def rotmat_to_quat(R):
        m = R
        t = np.trace(m)
        if t > 0:
            s = np.sqrt(t + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / (s + 1e-12)
            y = (m[0, 2] - m[2, 0]) / (s + 1e-12)
            z = (m[1, 0] - m[0, 1]) / (s + 1e-12)
        else:
            i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
            if i == 0:
                s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
                w = (m[2, 1] - m[1, 2]) / (s + 1e-12)
                x = 0.25 * s
                y = (m[0, 1] + m[1, 0]) / (s + 1e-12)
                z = (m[0, 2] + m[2, 0]) / (s + 1e-12)
            elif i == 1:
                s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
                w = (m[0, 2] - m[2, 0]) / (s + 1e-12)
                x = (m[0, 1] + m[1, 0]) / (s + 1e-12)
                y = 0.25 * s
                z = (m[1, 2] + m[2, 1]) / (s + 1e-12)
            else:
                s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
                w = (m[1, 0] - m[0, 1]) / (s + 1e-12)
                x = (m[0, 2] + m[2, 0]) / (s + 1e-12)
                y = (m[1, 2] + m[2, 1]) / (s + 1e-12)
                z = 0.25 * s
        q = np.array([w, x, y, z], dtype=np.float64)
        return q / (np.linalg.norm(q) + 1e-12)

    def quat_to_rotmat(q):
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ], dtype=np.float64)

    def slerp(q0, q1, t):
        q0 = q0 / (np.linalg.norm(q0) + 1e-12)
        q1 = q1 / (np.linalg.norm(q1) + 1e-12)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        dot = np.clip(dot, -1.0, 1.0)
        if dot > 0.9995:
            q = q0 + t * (q1 - q0)
            return q / (np.linalg.norm(q) + 1e-12)
        theta0 = np.arccos(dot)
        sin0 = np.sin(theta0)
        theta = theta0 * t
        s0 = np.sin(theta0 - theta) / (sin0 + 1e-12)
        s1 = np.sin(theta) / (sin0 + 1e-12)
        return s0 * q0 + s1 * q1

    def find_neighbors(ts_sorted, t):
        # ts_sorted: 单个 camera_id 的归一化时间序列（升序）
        if t <= ts_sorted[0]:
            return 0, 0, 0.0
        if t >= ts_sorted[-1]:
            n = len(ts_sorted) - 1
            return n, n, 0.0
        j = int(np.searchsorted(ts_sorted, t, side="right"))
        i0, i1 = j - 1, j
        t0, t1 = float(ts_sorted[i0]), float(ts_sorted[i1])
        a = (t - t0) / (t1 - t0 + 1e-12)
        return i0, i1, float(a)

    # ===== 4) 给定 (cid, t) 得到该真实相机在时刻 t 的 c2w (R, C) =====
    def pose_c2w_at_time(cid, t):
        idxs_sorted = camid_to_sorted_indices[cid]
        ts_sorted = camid_to_sorted_times[cid]

        i0, i1, a = find_neighbors(ts_sorted, t)
        idx0 = int(idxs_sorted[i0])
        idx1 = int(idxs_sorted[i1])

        cam0 = dataset.all_cam_params[idx0]
        cam1 = dataset.all_cam_params[idx1]

        # c2w：orientation(camera-to-world), position(camera center in world)
        R0 = np.array(cam0.orientation, np.float64)
        C0 = np.array(cam0.position,    np.float64)
        R1 = np.array(cam1.orientation, np.float64)
        C1 = np.array(cam1.position,    np.float64)

        q0 = rotmat_to_quat(R0)
        q1 = rotmat_to_quat(R1)
        q  = slerp(q0, q1, a)
        R  = quat_to_rotmat(q)
        C  = (1.0 - a) * C0 + a * C1
        return R, C, idx0  # idx0 用作内参模板

    # ===== 5) 构造 view_defs：不足则虚拟插值补齐 =====
    M = int(num_views_per_time)

    if len(valid_cam_ids) >= M:
        # 够用：从真实相机里选
        if random_sample_view:
            chosen_cam_ids = list(np.random.choice(valid_cam_ids, size=M, replace=False))
        else:
            pick = np.linspace(0, len(valid_cam_ids) - 1, M).round().astype(int)
            chosen_cam_ids = [valid_cam_ids[i] for i in pick]
        view_defs = [(c, c, 0.0) for c in chosen_cam_ids]
    else:
        # 不够：用虚拟视角补齐（当前你遇到的是 len==2）
        if len(valid_cam_ids) == 1:
            # 只有1个相机：只能复制；对渲染没新增视角，但至少不报错
            c = valid_cam_ids[0]
            view_defs = [(c, c, 0.0) for _ in range(M)]
        else:
            # 采用“在相机对之间插值”生成 M 个视角
            # 这里优先处理 len==2：直接在这两台之间取 alpha∈[0,1]
            if len(valid_cam_ids) == 2:
                ca, cb = valid_cam_ids[0], valid_cam_ids[1]
                if random_sample_view:
                    alphas = np.sort(np.random.uniform(0.0, 1.0, M))
                else:
                    alphas = np.linspace(0.0, 1.0, M)
                view_defs = [(ca, cb, float(a)) for a in alphas]
            else:
                # len>=3 但仍不足 M：随机选 pair + alpha 补齐
                # 先把真实相机都放进去，然后补
                base = [(c, c, 0.0) for c in valid_cam_ids]
                need = M - len(base)
                pairs = []
                for _ in range(need):
                    ca, cb = np.random.choice(valid_cam_ids, size=2, replace=False)
                    a = float(np.random.uniform(0.0, 1.0))
                    pairs.append((int(ca), int(cb), a))
                view_defs = base + pairs
    # ===== 6) 写回 CameraInfo：兼容 namedtuple / 普通 class =====
    def set_cam_RT(cam_info, R, T):
        if hasattr(cam_info, "_replace"):
            return cam_info._replace(R=R, T=T)
        # fallback: class
        cam_info.R = R
        cam_info.T = T
        return cam_info

    # ===== 7) 生成 cam_infos_videos =====
    cam_infos_videos = []
    uid = 0

    for (ca, cb, alpha_view) in view_defs:
        one_video_infos = []

        for frame_idx in range(N):
            t = float(render_times[frame_idx])  # 归一化时间 [0,1]

            Ra, Ca, idxa = pose_c2w_at_time(ca, t)
            Rb, Cb, idxb = pose_c2w_at_time(cb, t)

            # view 维度插值（在 c2w 空间）
            if ca == cb or alpha_view <= 1e-12:
                Rv, Cv, idx_template = Ra, Ca, idxa
            elif alpha_view >= 1.0 - 1e-12:
                Rv, Cv, idx_template = Rb, Cb, idxb
            else:
                q0 = rotmat_to_quat(Ra)
                q1 = rotmat_to_quat(Rb)
                qv = slerp(q0, q1, float(alpha_view))
                Rv = quat_to_rotmat(qv)
                Cv = (1.0 - float(alpha_view)) * Ca + float(alpha_view) * Cb
                idx_template = idxa  # 内参模板：一般两台相机一致；不一致也可固定为某一台

            # 转回 w2c：R = Rcw^T, T = -C @ R
            R_w2c = Rv.T
            T_w2c = - Cv @ R_w2c

            cam_info = dataset.load_pose_time_only(
                idx_template,
                uid=uid,
                time_override=t * float(maxtime)
            )
            cam_info = set_cam_RT(cam_info, R_w2c, T_w2c)
            one_video_infos.append(cam_info)

            uid += 1

        cam_infos_videos.append(one_video_infos)

    return cam_infos_videos


def readHyperDataInfos_gen_new(
    datadir,
    use_bg_points,
    eval,
    ratio=0.5,
    train_num_views_per_time=100, train_num_frame = 160, train_random_sample_view=True, train_random_sample_time=True,
    test_num_views_per_time=100, test_num_frame = 20, test_random_sample_view=False, test_random_sample_time=False
):
    train_cam_infos = Load_hyper_data(datadir, ratio, use_bg_points, split="train")
    test_cam_infos  = Load_hyper_data(datadir, ratio, use_bg_points, split="test")
    print(len(train_cam_infos.all_cam), len(train_cam_infos.all_time))
    
    # print("load finished")

    train_cam = format_hyper_data(Load_hyper_data(datadir, ratio, use_bg_points, split="train"), "train")
    # print("format finished")

    max_time = train_cam_infos.max_time
    
    print(f"max_time: {max_time}")
    train_cam_infos = generateCamerasFromHyper_gen_new(
        dataset=Load_hyper_data(datadir, ratio, use_bg_points, split="train"),
        maxtime=max_time,
        num_views_per_time=train_num_views_per_time,
        num_frame=train_num_frame,
        random_sample_view=train_random_sample_view,
        random_sample_time=train_random_sample_time
    )
    test_cam_infos = generateCamerasFromHyper_gen_new(
        dataset=Load_hyper_data(datadir, ratio, use_bg_points, split="test"),
        maxtime=max_time,
        num_views_per_time=test_num_views_per_time,
        num_frame=test_num_frame,
        random_sample_view=test_random_sample_view,
        random_sample_time=test_random_sample_time
    )
    

    # ✅ 生成你要的 cam_infos_videos 结构
    video_cam_infos = generateCamerasFromHyper_gen_new(
        dataset=Load_hyper_data(datadir, ratio, use_bg_points, split="test"),
        maxtime=max_time,
        num_views_per_time=test_num_views_per_time,
        num_frame=test_num_frame,
        random_sample_view=test_random_sample_view,
        random_sample_time=test_random_sample_time
    )

    ply_path = os.path.join(datadir, "points3D_downsample2.ply")
    pcd = fetchPly(ply_path)
    xyz = np.array(pcd.points)
    pcd = pcd._replace(points=xyz)

    nerf_normalization = getNerfppNorm(train_cam)
    # plot_camera_orientations(train_cam_infos, pcd.points)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        video_cameras=video_cam_infos,  # ✅ [ [CameraInfo]*N ]*M
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        maxtime=max_time
    )
    return scene_info


def readHyperDataInfos(datadir,use_bg_points,eval):
    train_cam_infos = Load_hyper_data(datadir,0.5,use_bg_points,split ="train")
    test_cam_infos = Load_hyper_data(datadir,0.5,use_bg_points,split="test")
    print("load finished")
    train_cam = format_hyper_data(train_cam_infos,"train")
    print("format finished")
    max_time = train_cam_infos.max_time
    video_cam_infos = copy.deepcopy(test_cam_infos)
    video_cam_infos.split="video"


    ply_path = os.path.join(datadir, "points3D_downsample2.ply")
    pcd = fetchPly(ply_path)
    xyz = np.array(pcd.points)

    pcd = pcd._replace(points=xyz)
    nerf_normalization = getNerfppNorm(train_cam)
    plot_camera_orientations(train_cam_infos, pcd.points)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time
                           )

    return scene_info


def format_render_poses(poses,data_infos):
    cameras = []
    tensor_to_pil = transforms.ToPILImage()
    len_poses = len(poses)
    times = [i/len_poses for i in range(len_poses)]
    image = data_infos[0][0]
    for idx, p in tqdm(enumerate(poses)):
        # image = None
        image_path = None
        image_name = f"{idx}"
        time = times[idx]
        pose = np.eye(4)
        pose[:3,:] = p[:3,:]
        # matrix = np.linalg.inv(np.array(pose))
        R = pose[:3,:3]
        R = - R
        R[:,0] = -R[:,0]
        T = -pose[:3,3].dot(R)
        FovX = focal2fov(data_infos.focal[0], image.shape[2])
        FovY = focal2fov(data_infos.focal[0], image.shape[1])
        cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                            time = time, mask=None))
    return cameras

def add_points(pointsclouds, xyz_min, xyz_max):
    add_points = (np.random.random((100000, 3)))* (xyz_max-xyz_min) + xyz_min
    add_points = add_points.astype(np.float32)
    addcolors = np.random.random((100000, 3)).astype(np.float32)
    addnormals = np.random.random((100000, 3)).astype(np.float32)
    # breakpoint()
    new_points = np.vstack([pointsclouds.points,add_points])
    new_colors = np.vstack([pointsclouds.colors,addcolors])
    new_normals = np.vstack([pointsclouds.normals,addnormals])
    pointsclouds=pointsclouds._replace(points=new_points)
    pointsclouds=pointsclouds._replace(colors=new_colors)
    pointsclouds=pointsclouds._replace(normals=new_normals)
    return pointsclouds
    # breakpoint()
    # new_
def readdynerfInfo(datadir,use_bg_points,eval):
    # loading all the data follow hexplane format
    # ply_path = os.path.join(datadir, "points3D_dense.ply")
    ply_path = os.path.join(datadir, "points3D_downsample2.ply")
    from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
    train_dataset = Neural3D_NDC_Dataset(
    datadir,
    "train",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=0,
        )    
    test_dataset = Neural3D_NDC_Dataset(
    datadir,
    "test",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=0,
        )
    train_cam_infos = format_infos(train_dataset,"train")
    val_cam_infos = format_render_poses(test_dataset.val_poses,test_dataset)
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # xyz = np.load
    pcd = fetchPly(ply_path)
    print("origin points,",pcd.points.shape[0])
    
    print("after points,",pcd.points.shape[0])

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_dataset,
                           test_cameras=test_dataset,
                           video_cameras=val_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=300
                           )
    return scene_info

def readdynerfInfo_gen_new(
    datadir,
    use_bg_points,
    eval,
    num_views_per_time=100,   # M
    num_frame=8,              # N
    random_sample_view=True,
    random_sample_time=False,
):
    import os
    import numpy as np
    from collections import defaultdict
    from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset

    # ---- 原始 dataset 加载 ----
    ply_path = os.path.join(datadir, "points3D_downsample2.ply")

    train_dataset = Neural3D_NDC_Dataset(
        datadir, "train", 1.0,
        time_scale=1,
        scene_bbox_min=[-2.5, -2.0, -1.0],
        scene_bbox_max=[2.5, 2.0, 1.0],
        eval_index=0,
    )
    test_dataset = Neural3D_NDC_Dataset(
        datadir, "test", 1.0,
        time_scale=1,
        scene_bbox_min=[-2.5, -2.0, -1.0],
        scene_bbox_max=[2.5, 2.0, 1.0],
        eval_index=0,
    )

    # 扁平 CameraInfo 列表（含 time 字段）
    train_flat = format_infos(train_dataset, "train")   # List[CameraInfo]
    test_flat  = format_infos(test_dataset,  "train")   # test split 也按同样格式化（如你有专门的 format_infos/test 可换）

    # 你原来的 val_cam_infos 是渲染poses，不一定有“每个 time 的多视角”，这里先不用它
    # val_cam_infos = format_render_poses(test_dataset.val_poses, test_dataset)

    # ---- 工具：按 time 分组 ----
    def group_by_time(cam_list):
        buckets = defaultdict(list)
        for c in cam_list:
            buckets[float(c.time)].append(c)
        times = sorted(buckets.keys())
        return buckets, times

    # ---- 工具：选 N 个时间点 ----
    def pick_times(times, N, random_sample):
        if len(times) == 0:
            raise RuntimeError("No cameras found (times empty). Check dynerf dataset formatting.")
        if N <= 0:
            raise ValueError(f"num_frame must be > 0, got {N}")
        if len(times) == 1:
            return [times[0]] * N

        if random_sample:
            idx = np.random.randint(0, len(times), size=N)
            chosen = [times[i] for i in sorted(idx)]
        else:
            idx = np.linspace(0, len(times) - 1, N)
            idx = np.round(idx).astype(int)
            chosen = [times[i] for i in idx]
        return chosen

    # ---- 工具：在某个 time 下选 M 个视角 ----
    def pick_views(cams_at_t, M, random_sample):
        cams = list(cams_at_t)
        if len(cams) == 0:
            raise RuntimeError("Empty camera list at some time bucket.")
        if M <= 0:
            raise ValueError(f"num_views_per_time must be > 0, got {M}")

        # 用 uid 排序保证稳定（也可以换 image_name）
        cams.sort(key=lambda x: x.uid)

        if len(cams) >= M:
            if random_sample:
                sel = np.random.choice(len(cams), size=M, replace=False)
                sel = sorted(sel)
                return [cams[i] for i in sel]
            else:
                idx = np.linspace(0, len(cams) - 1, M)
                idx = np.round(idx).astype(int)
                return [cams[i] for i in idx]
        else:
            # 不够 M 个相机：重复补齐
            if random_sample:
                sel = np.random.choice(len(cams), size=M, replace=True)
                return [cams[i] for i in sel]
            else:
                out = []
                i = 0
                while len(out) < M:
                    out.append(cams[i % len(cams)])
                    i += 1
                return out

    # ---- 核心：重组为 [M][N] ----
    def build_videos(flat_list, M, N, rand_view, rand_time):
        buckets, times = group_by_time(flat_list)
        chosen_times = pick_times(times, N, rand_time)

        videos = [[] for _ in range(M)]
        new_uid = 0

        for t in chosen_times:
            cams_at_t = buckets[t]
            chosen_views = pick_views(cams_at_t, M, rand_view)

            for m in range(M):
                c = chosen_views[m]
                # 复制一份，重新编号 uid，避免重复/冲突
                c2 = CameraInfo(
                    uid=new_uid,
                    R=c.R,
                    T=c.T,
                    FovY=c.FovY,
                    FovX=c.FovX,
                    image=c.image,
                    image_path=c.image_path,
                    image_name=c.image_name,
                    width=c.width,
                    height=c.height,
                    time=c.time,
                    mask=c.mask,
                )
                new_uid += 1
                videos[m].append(c2)

        return videos

    M = int(num_views_per_time)
    N = int(num_frame)

    train_cam_infos = build_videos(train_flat, M, N, random_sample_view, random_sample_time)
    test_cam_infos  = build_videos(test_flat,  M, N, random_sample_view, random_sample_time)

    # video：通常可以用 test_flat 或 train_flat 来生成
    video_cam_infos = build_videos(test_flat, M, N, random_sample_view, random_sample_time)

    # ---- normalization：与 Blender_gen_new 一致返回 list ----
    nerf_normalization = [getNerfppNorm(one_video) for one_video in train_cam_infos]

    # ---- 点云 ----
    pcd = fetchPly(ply_path)
    print("origin points,", pcd.points.shape[0])
    print("after points,",  pcd.points.shape[0])

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,   # ✅ [M][N]
        test_cameras=test_cam_infos,     # ✅ [M][N]
        video_cameras=video_cam_infos,   # ✅ [M][N]
        nerf_normalization=nerf_normalization,  # ✅ list
        ply_path=ply_path,
        maxtime=300
    )
    return scene_info



def setup_camera(w, h, k, w2c, near=0.01, far=100):
    from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        debug=True
    )
    return cam
def plot_camera_orientations(cam_list, xyz):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax2 = fig.add_subplot(122, projection='3d')
    # xyz = xyz[xyz[:,0]<1]
    threshold=2
    xyz = xyz[(xyz[:, 0] >= -threshold) & (xyz[:, 0] <= threshold) &
                         (xyz[:, 1] >= -threshold) & (xyz[:, 1] <= threshold) &
                         (xyz[:, 2] >= -threshold) & (xyz[:, 2] <= threshold)]

    ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c='r',s=0.1)
    for cam in tqdm(cam_list):
        # 提取 R 和 T
        R = cam.R
        T = cam.T

        direction = R @ np.array([0, 0, 1])

        ax.quiver(T[0], T[1], T[2], direction[0], direction[1], direction[2], length=1)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.savefig("output.png")
    # breakpoint()
def readPanopticmeta(datadir, json_path):
    with open(os.path.join(datadir,json_path)) as f:
        test_meta = json.load(f)
    w = test_meta['w']
    h = test_meta['h']
    max_time = len(test_meta['fn'])
    cam_infos = []
    for index in range(len(test_meta['fn'])):
        focals = test_meta['k'][index]
        w2cs = test_meta['w2c'][index]
        fns = test_meta['fn'][index]
        cam_ids = test_meta['cam_id'][index]

        time = index / len(test_meta['fn'])
        for focal, w2c, fn, cam in zip(focals, w2cs, fns, cam_ids):
            image_path = os.path.join(datadir,"ims")
            image_name=fn
            image = Image.open(os.path.join(datadir,"ims",fn))
            im_data = np.array(image.convert("RGBA"))
            im_data = PILtoTorch(im_data,None)[:3,:,:]
            camera = setup_camera(w, h, focal, w2c)
            cam_infos.append({
                "camera":camera,
                "time":time,
                "image":im_data})
            
    cam_centers = np.linalg.inv(test_meta['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    return cam_infos, max_time, scene_radius 

def readPanopticSportsinfos(datadir):
    train_cam_infos, max_time, scene_radius = readPanopticmeta(datadir, "train_meta.json")
    test_cam_infos,_, _ = readPanopticmeta(datadir, "test_meta.json")
    nerf_normalization = {
        "radius":scene_radius,
        "translate":torch.tensor([0,0,0])
    }

    ply_path = os.path.join(datadir, "pointd3D.ply")

        # Since this data set has no colmap data, we start with random points
    plz_path = os.path.join(datadir, "init_pt_cld.npz")
    data = np.load(plz_path)["data"]
    xyz = data[:,:3]
    rgb = data[:,3:6]
    num_pts = xyz.shape[0]
    pcd = BasicPointCloud(points=xyz, colors=rgb, normals=np.ones((num_pts, 3)))
    storePly(ply_path, xyz, rgb)
    # pcd = fetchPly(ply_path)
    # breakpoint()
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time,
                           )
    return scene_info

def readMultipleViewinfos(datadir,llffhold=8):

    cameras_extrinsic_file = os.path.join(datadir, "sparse_/images.bin")
    cameras_intrinsic_file = os.path.join(datadir, "sparse_/cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    from scene.multipleview_dataset import multipleview_dataset
    train_cam_infos = multipleview_dataset(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, cam_folder=datadir,split="train")
    test_cam_infos = multipleview_dataset(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, cam_folder=datadir,split="test")

    train_cam_infos_ = format_infos(train_cam_infos,"train")
    nerf_normalization = getNerfppNorm(train_cam_infos_)

    ply_path = os.path.join(datadir, "points3D_multipleview.ply")
    bin_path = os.path.join(datadir, "points3D_multipleview.bin")
    txt_path = os.path.join(datadir, "points3D_multipleview.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    
    try:
        pcd = fetchPly(ply_path)
        
    except:
        pcd = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=test_cam_infos.video_cam_infos,
                           maxtime=0,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Blender_gen_new": readNerfSyntheticInfo_gen_new,
    "dynerf" : readdynerfInfo,
    "nerfies": readHyperDataInfos,  # NeRFies & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
    "nerfies_gen_new": readHyperDataInfos_gen_new,
    "PanopticSports" : readPanopticSportsinfos,
    "MultipleView": readMultipleViewinfos
}
