# Import required libraries
import os  # filesystem operations
import json  # JSON file handling
import torch  # PyTorch deep learning framework
import numpy as np  # numerical computing library

# gaussian-splatting utils
import sys
sys.path.append(os.path.join(os.getcwd(), 'gaussian-splatting'))
# sys.path.insert(0, os.path.join(os.getcwd(), 'gaussian-splatting'))
from scene.cameras import MiniCam  # camera class definition
from utils.graphics_utils import focal2fov, getProjectionMatrix, fov2focal  # graphics utility functions
from PIL import Image  # image processing library

# Vector normalization: divide a vector by its magnitude to obtain a unit vector
normalize = lambda x : x / torch.linalg.norm(x)

# Translation matrix along the Z axis by distance t
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

# Rotation matrix about the X axis by angle phi (radians)
rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

# Rotation matrix about the Y axis by angle th (radians)
rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def get_SE3_from_SO3_and_Trans(r, t):
    """
    Build an SE(3) transform from a rotation matrix (SO(3)) and a translation vector.

    Args:
        r: rotation matrix, shape (N, 3, 3) or (N*3, 3)
        t: translation vector, shape (N, 3)

    Returns:
        SE3: SE(3) transform matrix, shape (N, 4, 4)
    """
    SE3 = torch.eye(4).repeat(len(r), 1, 1).cuda()
    SE3[:, :3, :3] = r if len(r.shape) == 3 else r.reshape(-1, 3, 3)
    SE3[:, :3, 3] = t
    return SE3

def get_SE3_from_m3x4(m3x4):
    """
    Build an SE(3) transform from a 3x4 matrix.

    Args:
        m3x4: 3x4 transform matrix, shape (N, 3, 4)

    Returns:
        SE3: SE(3) transform matrix, shape (N, 4, 4)
    """
    r, t = m3x4[:, :3, :3], m3x4[:, :3, 3]
    return get_SE3_from_SO3_and_Trans(r, t)

def viewmatrix(z, up, pos):
    """
    Build a view matrix (camera-to-world transform).

    Args:
        z: camera forward direction
        up: camera up direction
        pos: camera position

    Returns:
        m: 4x4 view matrix (c2w)
    """
    vec2 = normalize(z)  # normalized forward vector
    vec0 = normalize(torch.linalg.cross(up, vec2))  # right vector
    vec1 = normalize(torch.linalg.cross(vec2, vec0))  # up vector
    m = torch.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(c2ws):
    """
    Compute the average pose of multiple cameras.

    Args:
        c2ws: camera-to-world transforms, shape (N, 4, 4)

    Returns:
        c2w: average camera pose matrix
    """
    center = c2ws[:, :3, 3].mean(0)  # mean camera position
    vec2 = normalize(c2ws[:, :3, 2].sum(0))  # mean forward vector
    up = c2ws[:, :3, 1].sum(0)  # mean up vector
    return viewmatrix(vec2, up, center) #c2w

def recenter_poses(poses):
    """
    Recenter camera poses so that cameras are distributed around the origin.

    Args:
        poses: camera pose matrices, shape (N, 4, 4)

    Returns:
        poses_: recentered camera pose matrices
    """
    poses_ = torch.clone(poses)
    bottom = torch.tensor([0, 0, 0, 1])[None].cuda()
    c2w = poses_avg(poses)  # average pose
    c2w = torch.cat([c2w, bottom])
    poses = torch.cat([poses[:, :3, :4], bottom.repeat(len(poses), 1, 1)], 1)
    poses = torch.linalg.inv(c2w).repeat(len(poses), 1, 1).bmm(poses)
    poses_[:, :3, :4] = poses[:, :3, :4]
    return poses_

def pose_spherical(theta, phi, radius):
    """
    Generate a camera pose in spherical coordinates.

    Args:
        theta: azimuth angle (degrees)
        phi: elevation angle (degrees)
        radius: distance from the origin

    Returns:
        c2w: camera-to-world transform
    """
    c2w = trans_t(radius)  # translate along the Z axis
    c2w = rot_phi(phi / 180. * np.pi) @ c2w  # rotate about the X axis
    c2w = rot_theta(theta / 180. * np.pi) @ c2w  # rotate about the Y axis
    c2w = torch.Tensor(
        np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w  # coordinate-frame conversion
    return c2w

def get_data_infos(model_path, source_path, data_type):
    """
    Load basic dataset information (camera parameters, image size, and so on).

    Args:
        model_path: model path containing cameras.json
        source_path: source data path containing poses_bounds.npy
        data_type: dataset type, 'blender' or 'llff'

    Returns:
        dict: camera parameters and dataset information
    """
    # Load the camera parameter file
    with open(os.path.join(model_path, f'cameras.json')) as jf:
        clist = json.load(jf)

    # Read basic camera parameters
    width = clist[0]['width']
    height = clist[0]['height']
    fovx = focal2fov(clist[0]['fx'], width)  # horizontal field of view
    fovy = focal2fov(clist[0]['fy'], height)  # vertical field of view
    znear = 0.01  # near plane distance
    zfar = 100.0  # far plane distance

    if data_type == 'blender':
        # Blender dataset: return the basic parameters directly
        return {
            'width' : width,
            'height': height,
            'fovx'  : fovx,
            'fovy'  : fovy,
            'znear' : znear,
            'zfar'  : zfar
        }

    else:
        # LLFF dataset: additional pose processing is required
        Rs = torch.cat([torch.tensor(c['rotation'])[None] for c in clist]).cuda()
        Ts = torch.cat([torch.tensor(c['position'])[None] for c in clist]).cuda()
        w2cs = get_SE3_from_SO3_and_Trans(Rs, Ts)
        c2ws = torch.cat([torch.linalg.inv(w2c)[None] for w2c in w2cs])

        # Load pose bounds
        poses_arr = np.load(os.path.join(source_path, 'poses_bounds.npy'))
        bds = poses_arr[:, -2:].transpose([1, 0])
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)
        bd_factor = 0.75
        sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
        bds *= sc
        close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
        c2ws[:, :3, 3] *= sc

        # Recenter poses
        c2ws = recenter_poses(c2ws)
        c2w = poses_avg(c2ws)
        up = normalize(c2ws[:, :3, 1].sum(0))
        tt = c2ws[:, :3, 3]
        rads = torch.ones(4).cuda()
        rads[:3] = torch.quantile(torch.abs(tt), 0.9, 0)
        dt = 0.75
        focal = 1. / (((1. - dt) / close_depth + dt / inf_depth))
        
        return {
            'width' : width // 8,  # downsample for the LLFF dataset
            'height': height // 8,
            'fovx'  : fovx,
            'fovy'  : fovy,
            'znear' : znear,
            'zfar'  : zfar,
            'c2w'   : c2w,
            'up'    : up,
            'rads'  : rads,
            'focal' : float(focal)
        }
    
def get_supp_cameras(model_path, data_type):
    """
    Load the support camera list (original cameras used for watermarking).

    Args:
        model_path: model path containing cameras.json
        data_type: dataset type, 'blender' or 'llff'

    Returns:
        list: list of MiniCam objects
    """
    # Load the camera parameter file
    with open(os.path.join(model_path, f'cameras.json')) as jf:
        clist = json.load(jf)

    # Read basic camera parameters
    width = clist[0]['width']
    height = clist[0]['height']
    fovx = focal2fov(clist[0]['fx'], width)
    fovy = focal2fov(clist[0]['fy'], height)
    znear = 0.01
    zfar = 100.0

    # Set parameters according to the dataset type
    params = {
        'width' : width if data_type == 'blender' else width // 8,
        'height': height if data_type == 'blender' else height // 8,
        'fovx'  : fovx,
        'fovy'  : fovy,
        'znear' : znear,
        'zfar'  : zfar
    }

    # Build the transform matrix for each camera
    w2cs = []
    for c in clist:
        c2w = np.eye(4)
        rot = np.array(c['rotation'])
        pos = np.array(c['position'])
        c2w[:3, :3] = rot
        c2w[:3, 3] = pos
        w2c = np.linalg.inv(c2w).transpose()  # convert to a world-to-camera transform
        w2cs.append(w2c)
    
    w2cs = torch.from_numpy(np.array(w2cs)).cuda().float()
    proj_matrix = getProjectionMatrix(znear=params['znear'], zfar=params['zfar'], fovX=params['fovx'], fovY=params['fovy']).transpose(0,1).cuda()
    projs = w2cs.bmm(proj_matrix.repeat(len(w2cs), 1, 1))
    return [MiniCam(world_view_transform=w2c, full_proj_transform=proj, **params) for w2c, proj in zip(w2cs, projs)]

def get_cameras(num_cameras, data_type, data_infos, is_random=True):
    """
    Generate a given number of camera objects.

    Args:
        num_cameras: number of cameras to generate
        data_type: dataset type, 'blender' or 'llff'
        data_infos: dataset information dictionary
        is_random: whether to sample camera positions randomly

    Returns:
        list: list of MiniCam objects
    """
    # Dispatch to the pose generator for the given dataset type
    w2cs, params = globals()[f'get_{data_type}_poses'](num_cameras, data_infos, is_random)
    proj_matrix = getProjectionMatrix(znear=params['znear'], zfar=params['zfar'], fovX=params['fovx'], fovY=params['fovy']).transpose(0,1).cuda()
    projs = w2cs.bmm(proj_matrix.repeat(len(w2cs), 1, 1))
    return [MiniCam(world_view_transform=w2c, full_proj_transform=proj, **params) for w2c, proj in zip(w2cs, projs)]

def get_blender_poses(num_cameras, data_infos, is_random, theta_min=-180., theta_max=180., phi_min=-30., phi_max=-30., radius_min=4.031128874, radius_max=4.031128874):
    """
    Generate camera poses for the Blender dataset.

    Args:
        num_cameras: number of cameras to generate
        data_infos: dataset information dictionary
        is_random: whether to sample camera positions randomly
        theta_min, theta_max: azimuth range (degrees)
        phi_min, phi_max: elevation range (degrees)
        radius_min, radius_max: radius range

    Returns:
        tuple: (w2c transforms, camera parameter dictionary)
    """
    # Sample spherical coordinates
    thetas = np.random.uniform(theta_min, theta_max, num_cameras) if is_random else np.linspace(theta_min, theta_max, num_cameras)
    phis = np.random.uniform(phi_min, phi_max, num_cameras)
    radiuss = np.random.uniform(radius_min, radius_max, num_cameras)
    
    # Build camera-to-world transforms
    c2ws = torch.stack([pose_spherical(t, p, r) for t, p, r in zip(thetas, phis, radiuss)], 0)
    c2ws[:, :3, 1:3] *= -1  # flip the Y and Z axes
    
    # Convert to world-to-camera transforms
    return torch.cat([torch.linalg.inv(c2w).transpose(0, 1)[None].cuda() for c2w in c2ws]), data_infos

def get_llff_poses(num_cameras, data_infos, is_random):
    """
    Generate camera poses for the LLFF dataset.

    Args:
        num_cameras: number of cameras to generate
        data_infos: dataset information dictionary
        is_random: whether to sample camera positions randomly

    Returns:
        tuple: (w2c transforms, camera parameter dictionary)
    """
    # Extract key parameters from the dataset information
    c2w, up, rads, focal = [data_infos.pop(x) for x in ['c2w', 'up', 'rads', 'focal']]
    c2w = c2w.repeat(num_cameras, 1, 1)
    
    # Set camera trajectory parameters
    rots, zrate = 2, 0.5
    if is_random:
        # Sample angles randomly
        thetas = torch.rand(num_cameras) * 2 * np.pi * rots  
    else: 
        # Distribute angles uniformly
        thetas = torch.linspace(0., 2 * np.pi * rots, num_cameras + 1)[:-1]
    
    # Generate camera positions along a spiral trajectory
    xs = torch.cat([torch.tensor([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.])[None].cuda() * rads[None] for theta in thetas])
    cs = torch.bmm(c2w, xs[..., None]).squeeze(-1)
    fs = torch.tensor([0, 0, -focal, 1]).repeat(num_cameras, 1)[..., None].cuda()
    zs = cs - torch.bmm(c2w, fs).squeeze(-1)
    
    # Build view matrices
    m3x4s = torch.cat([viewmatrix(normalize(z), up, c)[None] for z, c in zip(zs, cs)])
    return torch.cat([torch.linalg.inv(c2w)[None] for c2w in get_SE3_from_m3x4(m3x4s)]), data_infos
