# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
import os
import os.path as osp
import timm
import torch
from opentome.timm import tome, dtem, diffrate, tofu, mctf, crossget, dct, pitome
from opentome.tome import tome as tm
import argparse
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import random
from typing import List, Tuple
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

torch.backends.cudnn.benchmark = True

try:
    from scipy.ndimage import binary_erosion
except ImportError:
    pass  # Don't fail if scipy is not installed. It's only necessary for this one file.


def generate_colormap(N: int, seed: int = 0) -> List[Tuple[float, float, float]]:
    """Generates a equidistant colormap with N elements."""
    random.seed(seed)

    def generate_color():
        return (random.random(), random.random(), random.random())

    return [generate_color() for _ in range(N)]


def make_visualization(
    img: Image, source: torch.Tensor, patch_size: int = 16, class_token: bool = True
) -> Image:
    """
    Create a visualization like in the paper.

    Args:
     -

    Returns:
     - A PIL image the same size as the input.
    """

    img = np.array(img.convert("RGB")) / 255.0
    source = source.detach().cpu()

    h, w, _ = img.shape
    ph = h // patch_size
    pw = w // patch_size

    if class_token:
        source = source[:, :, 1:]

    vis = source.argmax(dim=1)
    num_groups = vis.max().item() + 1

    cmap = generate_colormap(num_groups)
    vis_img = 0

    for i in range(num_groups):
        mask = (vis == i).float().view(1, 1, ph, pw)
        mask = F.interpolate(mask, size=(h, w), mode="nearest")
        mask = mask.view(h, w, 1).numpy()

        color = (mask * img).sum(axis=(0, 1)) / mask.sum()
        mask_eroded = binary_erosion(mask[..., 0])[..., None]
        mask_edge = mask - mask_eroded

        if not np.isfinite(color).all():
            color = np.zeros(3)

        vis_img = vis_img + mask_eroded * color.reshape(1, 1, 3)
        vis_img = vis_img + mask_edge * np.array(cmap[i]).reshape(1, 1, 3)

    # Convert back into a PIL image
    vis_img = Image.fromarray(np.uint8(vis_img * 255))

    return vis_img


def vis_eval(model, work_dir, args):
    input_size = model.module.default_cfg["input_size"][1]

    # Make sure the transform is correct for your model!
    transform_list = [
        transforms.Resize(int((256 / 224) * input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size)
    ]
    # The visualization and model need different transforms
    transform_vis  = transforms.Compose(transform_list)
    transform_norm = transforms.Compose(transform_list + [
        transforms.ToTensor(),
        transforms.Normalize(model.module.default_cfg["mean"], model.module.default_cfg["std"]),
    ])
    
    img = Image.open("./demo/n02510455_205.jpeg")
    vis = transform_vis(img)
    img = transform_norm(img)[None, ...]
    model.eval()
    _ = model(img)

    if args.save_vis:
        attn_source = model._tome_info["source"]
        if attn_source is None:
            raise ValueError("The model does not support ToMe visualization. Please use a model that supports ToMe visualization.")
        print(f"{attn_source.shape[1]} tokens at the end")
        vis = make_visualization(vis, attn_source, args.patch_size)
        vis.save(osp.join(work_dir, '{}_{}_tokens.png'.format(args.tome, attn_source.shape[1])))
        print("Visualization saved...")


def parse_args():
    parser = argparse.ArgumentParser(
        description='OpenToMe test (and eval) a model')
    # Baiscal parameters
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224', help='evaluation model name')
    parser.add_argument('--patch_size', type=int, default=16, help='model patch size')
    parser.add_argument('--tome', type=str, default='none', help='ToMe implementation to use, options: [tome, none]')
    parser.add_argument('--merge_num', type=int, default=98, help='the number of merge tokens')
    parser.add_argument('--merge_ratio', type=float, default=None, help='the ratio of merge tokens in per layers')
    parser.add_argument('--inflect', type=float, default=-0.5, help='the inflect of merge ratio, default: -0.5')
    parser.add_argument('--save_vis', type=bool, default=True, help='whether to save the visualization of the merge tokens')
    # Environment parameters
    parser.add_argument('--work_dir', type=str, default='work_dirs/visualization', help='the dir to save logs and models')
    parser.add_argument('--gpu-id', type=int, default=0, help='id of gpu to use ' '(only applicable to non-distributed testing)')
    parser.add_argument('--launcher', choices=['none', 'slurm', 'pytorch'], default='none', help='job launcher')
    parser.add_argument('--local_rank', help='set local_rank for torch.distributed.launch (torch<2.0.0)', type=int, default=0)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--port', type=int, default=29501, help='port only works when launcher=="slurm"')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        work_dir = osp.join(args.work_dir, 'eval_{}_by_{}'.format(args.model_name, args.tome))
    os.makedirs(work_dir, exist_ok=True)

    # distributed evaluation
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) >= 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    else:
        raise ValueError('Evaulation with a single process on 1 GPUs.')
    assert args.rank >= 0

    # build the model
    model = timm.create_model(args.model_name, pretrained=True)
    if model == None:
        raise ValueError(f"Model '{args.model_name}' could not be created.")
    if args.local_rank == 0:
        print(f'Model {args.model_name} loaded successfully., param count:{sum([m.numel() for m in model.parameters()])}')

    # setup distributed training
    model = model.to(args.device)
    if args.distributed:
        if args.local_rank == 0:
            print("Using native Torch DistributedDataParallel.")
        model = DDP(model, device_ids=[args.local_rank])

    assert args.merge_num >= 0, "Please specify a positive merge number."
    assert args.inflect in [-0.5, 1, 2], "Please specify a valid inflect value."
    if args.tome.lower() in ['tome', 'tofu', 'crossget', 'dct', "pitome"]:
        if args.tome == 'tome':
            tome.tome_apply_patch(model, trace_source=True)
        elif args.tome == 'tofu':
            tofu.tofu_apply_patch(model, trace_source=True)
        elif args.tome == 'crossget':
            crossget.crossget_apply_patch(model, trace_source=True)
        elif args.tome == 'dct':
            dct.dct_apply_patch(model, trace_source=True)
        elif args.tome == 'pitome':
            pitome.pitome_apply_patch(model, trace_source=True)
        if not hasattr(model, '_tome_info'):
            raise ValueError("The model does not support ToMe/ToFu/CrossGET. Please use a model that supports ToMe.")
        if args.merge_ratio is not None and args.merge_num is None:
            args.merge_num = sum(tm.parse_r(len(model.module.blocks), r=(args.merge_ratio, args.inflect)))
        elif args.merge_ratio is None and args.merge_num is not None:
            merge_ratio = tm.check_parse_r(len(model.module.blocks), args.merge_num, 
                                    (model.module.default_cfg["input_size"][1] / args.patch_size) ** 2, args.inflect)
        # update _tome_info
        model.r = (merge_ratio, args.inflect)
        model._tome_info["r"] = model.r
        model._tome_info["total_merge"] = args.merge_num
    elif args.tome.lower() == 'dtem':
        dtem.dtem_apply_patch(model, feat_dim=None)  # exteranal feature dim, defalut: none
        if not hasattr(model, '_tome_info'):
            raise ValueError("The model does not support DTEM. Please use a model that supports DTEM.")
        if args.merge_ratio is not None and args.merge_num is None:
            args.merge_num = sum(tm.parse_r(len(model.module.blocks), r=(args.merge_ratio, args.inflect)))
        elif args.merge_ratio is None and args.merge_num is not None:
            merge_ratio = tm.check_parse_r(len(model.module.blocks), args.merge_num, 
                                    (model.module.default_cfg["input_size"][1] / args.patch_size) ** 2, args.inflect)
        # update _tome_info
        model.r = (merge_ratio, args.inflect)
        model._tome_info["r"] = model.r
        model._tome_info["k2"] = 3
        model._tome_info["tau1"] = 0.1
        model._tome_info["tau2"] = 0.1
        model._tome_info["total_merge"] = args.merge_num
    elif args.tome.lower() == 'diffrate':
        diffrate.diffrate_apply_patch(model, prune_granularity=4, merge_granularity=4)
        if not hasattr(model, '_tome_info'):
            raise ValueError("The model does not support DiffRate. Please use a model that supports DiffRate.")
        r = args.merge_num / len(model.module.blocks) if args.merge_num is not None else 0
        model.init_kept_num_using_r(int(r))
    elif args.tome.lower() == 'mctf':
        mctf.mctf_apply_patch(model)
        if not hasattr(model, '_tome_info'):
            raise ValueError("The model does not support MCTF. Please use a model that supports MCTF.")
        if args.merge_ratio is not None and args.merge_num is None:
            args.merge_num = sum(tm.parse_r(len(model.module.blocks), r=(args.merge_ratio, args.inflect)))
        elif args.merge_ratio is None and args.merge_num is not None:
            merge_ratio = tm.check_parse_r(len(model.module.blocks), args.merge_num, 
                                    (model.module.default_cfg["input_size"][1] / args.patch_size) ** 2, args.inflect)
        # update _tome_info
        model.r = (merge_ratio, args.inflect)
        model._tome_info["r"] = model.r
        model._tome_info["total_merge"] = args.merge_num
        model._tome_info["one_step_ahead"] = 1
        model._tome_info["tau_sim"] = 1
        model._tome_info["tau_info"] = 20
        model._tome_info["tau_size"] = 40
        model._tome_info["bidirection"] = True
        model._tome_info["pooling_type"] = 'none'
    elif args.tome.lower() == 'none':
        pass
    else:
        raise ValueError("Invalid ToMe implementation specified. Use 'tome' or 'none'.")
    
    vis_eval(model, work_dir, args)


if __name__=="__main__":
    main()