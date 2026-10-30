import os
import sys
import random

import torch
from torch import nn
import torch.nn.functional as F
import clip
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
# gaussian-splatting utils
sys.path.append(os.path.join(os.getcwd(), '4DGaussians'))
# from lpipsPyTorch.modules.lpips import LPIPS
from lpips_pytorch.modules.lpips import LPIPS
lpips = LPIPS('alex', '0.1').cuda()
# lpips = LPIPS(net_type='alex').cuda()

# clip settings
CLIP_VERSION     = 'ViT-B/32'
CLIP_TOKEN_MIN   = 1
CLIP_TOKEN_MAX   = 49405
CLIP_TOKEN_BEGIN = 49406
CLIP_TOKEN_END   = 49407
CLIP_TOKEN_LEN   = 77
CLIP_TOKEN_NUM   = 75
CLIP_IMAGE_MEAN  = [0.48145466, 0.4578275, 0.40821073]
CLIP_IMAGE_STD   = [0.26862954, 0.26130258, 0.27577711]
CLIP_IMAGE_SIZE  = [224, 224]

class CLIPWatermarker(nn.Module):
    def __init__(self, **params):
        super().__init__()
        for k, v in params.items():
            self.__setattr__(k, v)

        self.image_mean = torch.tensor(CLIP_IMAGE_MEAN)[None, :, None, None]
        self.image_std = torch.tensor(CLIP_IMAGE_STD)[None, :, None, None]

        # loading clip model
        clip_model = clip.load(CLIP_VERSION)[0].float().eval()
        for k, v in clip_model.named_parameters():
            v.requires_grad = False

        self.clip_visual = clip_model.encode_image
        self.clip_text = clip_model.encode_text
        self.msg_decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, self.msg_len),
        )

    # token -> msg
    # def autoencoder(self, x, imgs):
    #     x = self.clip_text(x).float()
    #     imgs = self.resample(imgs)
    #     imgs = (imgs - self.image_mean.to(imgs.device)) / self.image_std.to(imgs.device)
        
    #     v_feats = self.clip_visual(imgs).float()
    #     input_feats = 0.5 * x + 0.5 * v_feats
    #     return self.decode(input_feats)
    
    def autoencoder(self, x, imgs, mix_ratio=0.5):
        x_text = self.clip_text(x).float()
        imgs = self.resample(imgs)
        imgs = (imgs - self.image_mean.to(imgs.device)) / self.image_std.to(imgs.device)
        v_feats = self.clip_visual(imgs).float()
        # mix_ratio 表示图像内容特征占比
        # mix_ratio=0.0: 纯消息特征
        # mix_ratio=0.5: 消息/图像各一半
        # mix_ratio=0.9: 图像干扰很强
        input_feats = (1.0 - mix_ratio) * x_text + mix_ratio * v_feats

        return self.decode(input_feats)

    # img -> msg
    def forward(self, x):
        if len(x.shape) == 5:
            feats = []
            for i in range(x.shape[0]):
                img = x[i]
                img = self.resample(img)
                img = (img - self.image_mean.to(img.device)) / self.image_std.to(img.device)
                feat = self.clip_visual(img).float()
                feat = feat.mean(dim=0)
                feats.append(feat)
            x = torch.stack(feats, dim=0)
        else:
            x = self.resample(x)
            x = (x - self.image_mean.to(x.device)) / self.image_std.to(x.device)
            x = self.clip_visual(x).float()
        return self.decode(x)
    
    def decode(self, x):
        x = x / x.norm(dim=1, keepdim=True)
        return self.msg_decoder(x)

    def resample(self, x):
        return x if x.shape[-2:] == CLIP_IMAGE_SIZE else F.interpolate(x, size=CLIP_IMAGE_SIZE, mode='bilinear', align_corners=True, antialias=True)


class MsgDataset_divid(Dataset):
    def __init__(self, **params):
        for k, v in params.items():
            self.__setattr__(k, v)

        self.all_imgs = []
        
        
        used_datasets = [
            "dnerf", #4d
            "hypernerf", #4d
            # "nerf_llff_data", #3d
            # "nerf_synthetic", #3d
        ]
        
        if "dnerf" in used_datasets:
        
            #4D dataset: dnerf
            sub_imgs = []
            img_root = "/workspace/syy/code/20251028_4Dwater/4Dwatermarker/4DGaussians/data/dnerf"
            senses = os.listdir(img_root)
            for sense in senses:
                imgs = os.listdir(os.path.join(img_root, sense, "train"))
                for img in imgs:
                    if img.endswith(".png"):
                        sub_imgs.append(os.path.join(img_root, sense, "train", img))
            print(f"collect {len(sub_imgs)} images from 4d dnerf")
            self.all_imgs.extend(sub_imgs)
        
        if "hypernerf" in used_datasets:
            #4D dataset: hypernerf
            sub_imgs = []
            img_root = "/workspace/syy/code/20251028_4Dwater/4Dwatermarker_umt/4DGaussians/data/hypernerf/virg"
            senses = [
                "3dprinter",
                "banana",
                "broom2",
                "chicken"
            ]
            for sense in senses:
                image_path = os.path.join(img_root, sense, "images")
                imgs = os.listdir(image_path)
                for img in imgs:
                    if img.endswith(".png"):
                        sub_imgs.append(os.path.join(image_path, img))
            print(f"collect {len(sub_imgs)} images from 4d hypernerf")
            self.all_imgs.extend(sub_imgs)
        
        if "nerf_llff_data" in used_datasets:
            #3d dataset: nerf_llff_data
            sub_imgs = []
            img_root = "/workspace/syy/code/20251028_4Dwater/GuardSplat/gaussian-splatting/data/nerf_llff_data"
            scenes = os.listdir(img_root)
            for scene in scenes:
                if not os.path.isdir(os.path.join(img_root, scene)):
                    continue
                image_path = os.path.join(img_root, scene, "images_8")
                imgs = os.listdir(image_path)
                for img in imgs:
                    if img.endswith(".png"):
                        sub_imgs.append(os.path.join(image_path, img))
            print(f"collect {len(sub_imgs)} images from 3d nerf_llff_data")
            self.all_imgs.extend(sub_imgs)
        
        if "nerf_synthetic" in used_datasets:
            #3d dataset: nerf_synthetic
            sub_imgs = []
            img_root = "/workspace/syy/code/20251028_4Dwater/GuardSplat/gaussian-splatting/data/nerf_synthetic"
            scenes = os.listdir(img_root)
            for scene in scenes:
                if not os.path.isdir(os.path.join(img_root, scene)):
                    continue
                image_path = os.path.join(img_root, scene, "train")
                imgs = os.listdir(image_path)
                for img in imgs:
                    if img.endswith(".png"):
                        sub_imgs.append(os.path.join(image_path, img))
            print(f"collect {len(sub_imgs)} images from 3d nerf_synthetic")
            self.all_imgs.extend(sub_imgs)
        
        random.shuffle(self.all_imgs)
        print(f"collect {len(self.all_imgs)} images")        
        self.load_data()
        self.image_mean = torch.tensor(CLIP_IMAGE_MEAN)[None, :, None, None]
        self.image_std = torch.tensor(CLIP_IMAGE_STD)[None, :, None, None]

    def __getitem__(self, index):
        img_path = random.choice(self.all_imgs)
        # img_path = self.all_imgs[index % len(self.all_imgs)]
        img = Image.open(img_path).convert("RGB")
        # img = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0)

        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        img = self.resample(img).squeeze(0)

        # img = (img - self.image_mean.to(img.device)) / self.image_std.to(img.device)
        # img = img.squeeze(0)
        return self.data[index] + (img,)
    
    def resample(self, x):
        return x if x.shape[-2:] == CLIP_IMAGE_SIZE else F.interpolate(x, size=CLIP_IMAGE_SIZE, mode='bilinear', align_corners=True, antialias=True)

    def __len__(self):
        return len(self.data)
    
    def load_data(self):
        max_value = int(2 ** self.msg_len)
        if not hasattr(self, 'b2t_maps'):
            rand_tokens = torch.randperm(CLIP_TOKEN_MAX) + 1
            self.b2t_maps = rand_tokens[:2 * self.msg_len].reshape(self.msg_len, 2)

        if not hasattr(self, 'data'):
            dec_messages = sample_K_from_N(self.max_size, max_value)
            self.data = [self.dec_message_to_tokens(x) for x in dec_messages]
        print(f"load {len(self.data)} data")

    # Decimal message -> Binary message + CLIP tokens
    def dec_message_to_tokens(self, dec_message):
        # Decimal message -> Binary message
        bin_message_text = bin(dec_message)[2:].zfill(self.msg_len)
        bin_message = torch.tensor([int(x) for x in bin_message_text])
        
        # Binary message -> Tokens
        tokens = torch.zeros(CLIP_TOKEN_LEN)
        tokens[0] = CLIP_TOKEN_BEGIN
        tokens[self.msg_len + 1] = CLIP_TOKEN_END
        for idx, (bit, b2t_map )in enumerate(zip(bin_message, self.b2t_maps)):
            tokens[idx + 1] = b2t_map[bit]

        return tokens.long(), bin_message.float()
    
    def get_params(self):
        return {k : getattr(self, k) for k in ['data', 'b2t_maps']}

class MsgDataset(Dataset):

    def __init__(self, **params):
        for k, v in params.items():
            self.__setattr__(k, v)

        self.load_data()

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def load_data(self):
        max_value = int(2 ** self.msg_len)
        if not hasattr(self, 'b2t_maps'):
            rand_tokens = torch.randperm(CLIP_TOKEN_MAX) + 1
            self.b2t_maps = rand_tokens[:2 * self.msg_len].reshape(self.msg_len, 2)

        if not hasattr(self, 'data'):
            dec_messages = sample_K_from_N(self.max_size, max_value)
            self.data = [self.dec_message_to_tokens(x) for x in dec_messages]

    # Decimal message -> Binary message + CLIP tokens
    def dec_message_to_tokens(self, dec_message):
        # Decimal message -> Binary message
        bin_message_text = bin(dec_message)[2:].zfill(self.msg_len)
        bin_message = torch.tensor([int(x) for x in bin_message_text])
        
        # Binary message -> Tokens
        tokens = torch.zeros(CLIP_TOKEN_LEN)
        tokens[0] = CLIP_TOKEN_BEGIN
        tokens[self.msg_len + 1] = CLIP_TOKEN_END
        for idx, (bit, b2t_map )in enumerate(zip(bin_message, self.b2t_maps)):
            tokens[idx + 1] = b2t_map[bit]

        return tokens.long(), bin_message.float()
    
    def get_params(self):
        return {k : getattr(self, k) for k in ['data', 'b2t_maps']}
    
# sample K Decimal messages from N population
def sample_K_from_N(K, N, Limit=48):

    def sample_from_large_population(K, N):
        rets = set()
        while len(rets) < K:
            tmp = random.randint(1, N)
            rets.add(tmp)

        return list(rets)

    if K >= N: return list(range(N))
    elif N <= 2 ** Limit: return random.sample(range(1, N), K)
    else: return sample_from_large_population(K, N)

@torch.no_grad()
def bit_accuracy(output, target):
    output = torch.where(output > 0.0, 1.0, 0.0)
    err = torch.logical_xor(output, target).sum().item() / target.numel()
    return (1. - err) * 100

def load_decoder_and_message(msg_len, sdir, seed=42):
    sdict = torch.load(os.path.join(sdir, f'CLIP-MsgDecoder-{msg_len}.pkl'),
                       map_location='cpu', weights_only=True)
    model = CLIPWatermarker(msg_len=msg_len)
    model.load_state_dict(sdict.pop('model'))
    for k, v in model.named_parameters():
        v.requires_grad = False
    # message = random.choice(sdict['data'])[1]
    py_rng = random.Random(seed)
    message = py_rng.choice(sdict['data'])[1]
    return model.cuda(), message.cuda()