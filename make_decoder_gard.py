import os
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import trange

from ms_utils_gard import CLIPWatermarker, MsgDataset, bit_accuracy

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


import random
import numpy as np
def seed_everything(seed: int = 0, deterministic: bool = True):
    # 1) Python / Numpy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 2) PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多卡

    # 3) cuDNN / 算法确定性
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch >= 1.8：更强的确定性约束（遇到不支持的算子会报错）
        torch.use_deterministic_algorithms(True)

        # CUDA 10.2+：让某些 CUDA 算子确定性（尤其是 matmul 类）
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)

# seed
seed_everything(0, deterministic=True)


def single_step_inference(batch, model, iscpu=False):
    input, target = [x.to(DEVICE) for x in batch]
    output = model.autoencoder(input)
    return (output.float().cpu(), target.cpu()) if iscpu else (output.float(), target)

def train(args):
    print (f'training decoder with N_L={args.msg_len}')
    dataset = MsgDataset(msg_len=args.msg_len, max_size=args.max_size)
    # trainloader = DataLoader(dataset, batch_size=min(args.batch_size, dataset.__len__()), shuffle=True)
    trainloader = DataLoader(dataset, batch_size=min(args.batch_size, dataset.__len__()), shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(dataset, batch_size=min(args.batch_size, dataset.__len__()), shuffle=False, num_workers=args.num_workers)
    model = CLIPWatermarker(msg_len=args.msg_len).to(DEVICE)
    optimizer = optim.Adam(model.msg_decoder.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    
    ####################################
    if args.save:
        os.makedirs(args.sdir, exist_ok=True)
    log_path = os.path.join(args.sdir, f'train_msglen{args.msg_len}.txt')
    with open(log_path, 'w', buffering=1) as f:
        f.write('epoch\tbacc\tloss\n')
    ####################################
    
        all_epochs = trange(1, args.num_epochs + 1)

        for eIdx in all_epochs:
            loss_value = 0
            for batch in trainloader:
                optimizer.zero_grad()
                loss = loss_fn(*single_step_inference(batch, model))
                loss.backward()
                optimizer.step()
                loss_value += loss.item()

            score = test(args, model, testloader)
            model.train()

            all_epochs.set_description(f'BAcc : {score} | Loss : {loss_value / dataset.__len__()}')
            
            # 只新增这一行：每轮写入 txt
            f.write(f'{eIdx}\t{score}\t{loss_value / dataset.__len__()}\n')

    if args.save:
        os.makedirs(args.sdir, exist_ok=True)
        sdict = {
            'model'   : model.eval().cpu().state_dict(),
            **{key : getattr(dataset, key) for key in ['data', 'b2t_maps']}
        }
        torch.save(sdict, os.path.join(args.sdir, f'CLIP-MsgDecoder-{args.msg_len}.pkl'))

def test(args, model=None, testloader=None):
    if model is None:
        print (f'testing decoder with N_L={args.msg_len}')
        sdict = torch.load(os.path.join(args.sdir, f'CLIP-MsgDecoder-{args.msg_len}.pkl'), map_location='cpu', weights_only=True)
        model = CLIPWatermarker(msg_len=args.msg_len)
        model.load_state_dict(sdict.pop('model'))
        model.to(DEVICE)
        dataset = MsgDataset(msg_len=args.msg_len, max_size=args.max_size, **sdict)
        testloader = DataLoader(dataset, batch_size=min(args.batch_size, dataset.__len__()), shuffle=False, num_workers=args.num_workers)

    with torch.no_grad():
        model.eval()
        outputs, targets = list(zip(*[single_step_inference(batch, model, iscpu=True) for batch in testloader]))
        outputs, targets = torch.cat(outputs), torch.cat(targets)

    if args.mode == 'train':
        return bit_accuracy(outputs, targets)
        
    else:
        print (f'BAcc : {bit_accuracy(outputs, targets):.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], required=True)
    parser.add_argument('--msg_len', type=int, choices=[4, 8, 16, 32, 48, 64, 72, 96, 128, 160, 192, 224, 256])
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--sdir', type=str, default='decoders/gard')
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_size', type=int, default=2048)
    parser.add_argument('--save', action='store_true') # whether saving ckpt file
    args = parser.parse_args()

    globals()[args.mode](args)