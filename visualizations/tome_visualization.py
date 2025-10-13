# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.

# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.
# # --------------------------------------------------------
# import os
# import os.path as osp
# import timm
# import torch
# from opentome.timm import tome, dtem, diffrate, tofu, mctf, crossget, dct, pitome
# from opentome.tome import tome as tm
# import argparse
# from torchvision import transforms
# from torchvision.transforms.functional import InterpolationMode
# from PIL import Image
# import random
# from typing import List, Tuple
# import numpy as np
# import torch.nn.functional as F
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP

# torch.backends.cudnn.benchmark = True

# try:
#     from scipy.ndimage import binary_erosion
# except ImportError:
#     pass  # Don't fail if scipy is not installed. It's only necessary for this one file.


# def generate_colormap(N: int, seed: int = 0) -> List[Tuple[float, float, float]]:
#     """Generates a equidistant colormap with N elements."""
#     random.seed(seed)

#     def generate_color():
#         return (random.random(), random.random(), random.random())

#     return [generate_color() for _ in range(N)]


# def make_visualization(
#     img: Image, source: torch.Tensor, patch_size: int = 16, class_token: bool = True
# ) -> Image:
#     """
#     Create a visualization like in the paper.

#     Args:
#      -

#     Returns:
#      - A PIL image the same size as the input.
#     """

#     img = np.array(img.convert("RGB")) / 255.0
#     source = source.detach().cpu()

#     h, w, _ = img.shape
#     ph = h // patch_size
#     pw = w // patch_size

#     if class_token:
#         source = source[:, :, 1:]

#     vis = source.argmax(dim=1)
#     num_groups = vis.max().item() + 1

#     cmap = generate_colormap(num_groups)
#     vis_img = 0

#     for i in range(num_groups):
#         mask = (vis == i).float().view(1, 1, ph, pw)
#         mask = F.interpolate(mask, size=(h, w), mode="nearest")
#         mask = mask.view(h, w, 1).numpy()

#         color = (mask * img).sum(axis=(0, 1)) / mask.sum()
#         mask_eroded = binary_erosion(mask[..., 0])[..., None]
#         mask_edge = mask - mask_eroded

#         if not np.isfinite(color).all():
#             color = np.zeros(3)

#         vis_img = vis_img + mask_eroded * color.reshape(1, 1, 3)
#         vis_img = vis_img + mask_edge * np.array(cmap[i]).reshape(1, 1, 3)

#     # Convert back into a PIL image
#     vis_img = Image.fromarray(np.uint8(vis_img * 255))

#     return vis_img


# def vis_eval(model, work_dir, args):
#     input_size = model.module.default_cfg["input_size"][1]

#     # Make sure the transform is correct for your model!
#     transform_list = [
#         transforms.Resize(int((256 / 224) * input_size), interpolation=InterpolationMode.BICUBIC),
#         transforms.CenterCrop(input_size)
#     ]
#     # The visualization and model need different transforms
#     transform_vis  = transforms.Compose(transform_list)
#     transform_norm = transforms.Compose(transform_list + [
#         transforms.ToTensor(),
#         transforms.Normalize(model.module.default_cfg["mean"], model.module.default_cfg["std"]),
#     ])
    
#     img = Image.open("./demo/n02510455_205.jpeg")
#     vis = transform_vis(img)
#     img = transform_norm(img)[None, ...]
#     model.eval()
#     _ = model(img)

#     if args.save_vis:
#         attn_source = model._tome_info["source"]
#         if attn_source is None:
#             raise ValueError("The model does not support ToMe visualization. Please use a model that supports ToMe visualization.")
#         print(f"{attn_source.shape[1]} tokens at the end")
#         vis = make_visualization(vis, attn_source, args.patch_size)
#         vis.save(osp.join(work_dir, '{}_{}_tokens.png'.format(args.tome, attn_source.shape[1])))
#         print("Visualization saved...")


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='OpenToMe test (and eval) a model')
#     # Baiscal parameters
#     parser.add_argument('--model_name', type=str, default='vit_base_patch16_224', help='evaluation model name')
#     parser.add_argument('--patch_size', type=int, default=16, help='model patch size')
#     parser.add_argument('--tome', type=str, default='none', help='ToMe implementation to use, options: [tome, none]')
#     parser.add_argument('--merge_num', type=int, default=98, help='the number of merge tokens')
#     parser.add_argument('--merge_ratio', type=float, default=None, help='the ratio of merge tokens in per layers')
#     parser.add_argument('--inflect', type=float, default=-0.5, help='the inflect of merge ratio, default: -0.5')
#     parser.add_argument('--save_vis', type=bool, default=True, help='whether to save the visualization of the merge tokens')
#     # Environment parameters
#     parser.add_argument('--work_dir', type=str, default='work_dirs/visualization', help='the dir to save logs and models')
#     parser.add_argument('--gpu-id', type=int, default=0, help='id of gpu to use ' '(only applicable to non-distributed testing)')
#     parser.add_argument('--launcher', choices=['none', 'slurm', 'pytorch'], default='none', help='job launcher')
#     parser.add_argument('--local_rank', help='set local_rank for torch.distributed.launch (torch<2.0.0)', type=int, default=0)
#     parser.add_argument('--local-rank', type=int, default=0)
#     parser.add_argument('--port', type=int, default=29501, help='port only works when launcher=="slurm"')
#     args = parser.parse_args()
#     if 'LOCAL_RANK' not in os.environ:
#         os.environ['LOCAL_RANK'] = str(args.local_rank)

#     return args


# def main():
#     args = parse_args()

#     # work_dir is determined in this priority: CLI > segment in file > filename
#     if args.work_dir is not None:
#         work_dir = osp.join(args.work_dir, 'eval_{}_by_{}'.format(args.model_name, args.tome))
#     os.makedirs(work_dir, exist_ok=True)

#     # distributed evaluation
#     args.distributed = False
#     if 'WORLD_SIZE' in os.environ:
#         args.distributed = int(os.environ['WORLD_SIZE']) >= 1
#     args.device = 'cuda:0'
#     args.world_size = 1
#     args.rank = 0  # global rank
#     if args.distributed:
#         args.device = 'cuda:%d' % args.local_rank
#         torch.cuda.set_device(args.local_rank)
#         torch.distributed.init_process_group(backend='nccl', init_method='env://')
#         args.world_size = torch.distributed.get_world_size()
#         args.rank = torch.distributed.get_rank()
#     else:
#         raise ValueError('Evaulation with a single process on 1 GPUs.')
#     assert args.rank >= 0

#     # build the model
#     model = timm.create_model(args.model_name, pretrained=True)
#     if model == None:
#         raise ValueError(f"Model '{args.model_name}' could not be created.")
#     if args.local_rank == 0:
#         print(f'Model {args.model_name} loaded successfully., param count:{sum([m.numel() for m in model.parameters()])}')

#     # setup distributed training
#     model = model.to(args.device)
#     if args.distributed:
#         if args.local_rank == 0:
#             print("Using native Torch DistributedDataParallel.")
#         model = DDP(model, device_ids=[args.local_rank])

#     assert args.merge_num >= 0, "Please specify a positive merge number."
#     assert args.inflect in [-0.5, 1, 2], "Please specify a valid inflect value."
#     if args.tome.lower() in ['tome', 'tofu', 'crossget', 'dct', "pitome"]:
#         if args.tome == 'tome':
#             tome.tome_apply_patch(model, trace_source=True)
#         elif args.tome == 'tofu':
#             tofu.tofu_apply_patch(model, trace_source=True)
#         elif args.tome == 'crossget':
#             crossget.crossget_apply_patch(model, trace_source=True)
#         elif args.tome == 'dct':
#             dct.dct_apply_patch(model, trace_source=True)
#         elif args.tome == 'pitome':
#             pitome.pitome_apply_patch(model, trace_source=True)
#         if not hasattr(model, '_tome_info'):
#             raise ValueError("The model does not support ToMe/ToFu/CrossGET. Please use a model that supports ToMe.")
#         if args.merge_ratio is not None and args.merge_num is None:
#             args.merge_num = sum(tm.parse_r(len(model.module.blocks), r=(args.merge_ratio, args.inflect)))
#         elif args.merge_ratio is None and args.merge_num is not None:
#             merge_ratio = tm.check_parse_r(len(model.module.blocks), args.merge_num, 
#                                     (model.module.default_cfg["input_size"][1] / args.patch_size) ** 2, args.inflect)
#         # update _tome_info
#         model.r = (merge_ratio, args.inflect)
#         model._tome_info["r"] = model.r
#         model._tome_info["total_merge"] = args.merge_num
#     elif args.tome.lower() == 'dtem':
#         dtem.dtem_apply_patch(model, feat_dim=None)  # exteranal feature dim, defalut: none
#         if not hasattr(model, '_tome_info'):
#             raise ValueError("The model does not support DTEM. Please use a model that supports DTEM.")
#         if args.merge_ratio is not None and args.merge_num is None:
#             args.merge_num = sum(tm.parse_r(len(model.module.blocks), r=(args.merge_ratio, args.inflect)))
#         elif args.merge_ratio is None and args.merge_num is not None:
#             merge_ratio = tm.check_parse_r(len(model.module.blocks), args.merge_num, 
#                                     (model.module.default_cfg["input_size"][1] / args.patch_size) ** 2, args.inflect)
#         # update _tome_info
#         model.r = (merge_ratio, args.inflect)
#         model._tome_info["r"] = model.r
#         model._tome_info["k2"] = 3
#         model._tome_info["tau1"] = 0.1
#         model._tome_info["tau2"] = 0.1
#         model._tome_info["total_merge"] = args.merge_num
#     elif args.tome.lower() == 'diffrate':
#         diffrate.diffrate_apply_patch(model, prune_granularity=4, merge_granularity=4)
#         if not hasattr(model, '_tome_info'):
#             raise ValueError("The model does not support DiffRate. Please use a model that supports DiffRate.")
#         r = args.merge_num / len(model.module.blocks) if args.merge_num is not None else 0
#         model.init_kept_num_using_r(int(r))
#     elif args.tome.lower() == 'mctf':
#         mctf.mctf_apply_patch(model)
#         if not hasattr(model, '_tome_info'):
#             raise ValueError("The model does not support MCTF. Please use a model that supports MCTF.")
#         if args.merge_ratio is not None and args.merge_num is None:
#             args.merge_num = sum(tm.parse_r(len(model.module.blocks), r=(args.merge_ratio, args.inflect)))
#         elif args.merge_ratio is None and args.merge_num is not None:
#             merge_ratio = tm.check_parse_r(len(model.module.blocks), args.merge_num, 
#                                     (model.module.default_cfg["input_size"][1] / args.patch_size) ** 2, args.inflect)
#         # update _tome_info
#         model.r = (merge_ratio, args.inflect)
#         model._tome_info["r"] = model.r
#         model._tome_info["total_merge"] = args.merge_num
#         model._tome_info["one_step_ahead"] = 1
#         model._tome_info["tau_sim"] = 1
#         model._tome_info["tau_info"] = 20
#         model._tome_info["tau_size"] = 40
#         model._tome_info["bidirection"] = True
#         model._tome_info["pooling_type"] = 'none'
#     elif args.tome.lower() == 'none':
#         pass
#     else:
#         raise ValueError("Invalid ToMe implementation specified. Use 'tome' or 'none'.")
    
#     vis_eval(model, work_dir, args)


# if __name__=="__main__":
#     main()

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
from PIL import Image, ImageDraw
import random
from typing import List, Tuple, Optional
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from opentome.tome.tome import token_unmerge

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


def make_attention_mask(
    img: Image, 
    mask: torch.tensor = None,
    attention_scores: torch.tensor = None, 
    patch_size: int = 16, 
    topk_ratio: float = 0.3,
    class_token: bool = True
) -> Image:
    """
    Create an attention mask visualization based on attention scores.
    
    Args:
        img: Original PIL image
        attention_scores: Attention scores tensor of shape (1, num_tokens) or (1, num_heads, num_tokens)
        patch_size: Size of each patch
        topk_ratio: Ratio of tokens to keep (0-1)
        class_token: Whether the first token is a class token
        
    Returns:
        A PIL image with attention mask overlay
    """
    # Process attention scores
    if attention_scores is not None:
        if attention_scores.dim() == 3:  # Multi-head attention
            # Average across heads
            attention_scores = attention_scores.mean(dim=1)
    
        # Remove class token if present
        if class_token:
            attention_scores = attention_scores[:, 1:]
        
        # Flatten and get top-k indices
        scores = attention_scores[0]  # Remove batch dimension
        print(scores.shape)
        num_tokens = scores.shape[0]
        k = int(topk_ratio * num_tokens)
        
        # Get top-k indices
        topk_values, topk_indices = torch.topk(scores, 124, sorted=False)

        mask = torch.zeros(num_tokens, dtype=torch.bool)
        mask[topk_indices] = True
    
    # Convert to numpy for visualization
    mask = mask.cpu().numpy()
    
    # Reshape mask to 2D grid
    img_size = img.size
    grid_size = (img_size[1] // patch_size, img_size[0] // patch_size)
    mask_2d = mask.reshape(grid_size)
    
    # Create overlay image
    overlay = Image.new('RGBA', img_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Draw rectangles for each patch
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if mask_2d[i, j]:
                x1 = j * patch_size
                y1 = i * patch_size
                x2 = x1 + patch_size
                y2 = y1 + patch_size
                draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 100))  # Semi-transparent red
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=1)
    
    # Composite with original image
    result = Image.alpha_composite(img.convert('RGBA'), overlay)
    return result.convert('RGB')


def make_visualization(
    img: Image, 
    source: torch.Tensor, 
    patch_size: int = 16, 
    class_token: bool = True,
    attention_scores: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    topk_ratio: float = 0.3
) -> Image:
    """
    Create a visualization like in the paper, with optional attention mask.

    Args:
        img: Input PIL image
        source: Token source information from ToMe
        patch_size: Size of each patch
        class_token: Whether the first token is a class token
        attention_scores: Optional attention scores for creating attention mask
        topk_ratio: Ratio of tokens to keep when using attention scores

    Returns:
        A PIL image the same size as the input.
    """
    # If attention scores are provided, create attention mask visualization
    if attention_scores is not None:
        return make_attention_mask(img, attention_scores=attention_scores, patch_size=patch_size, topk_ratio=topk_ratio, class_token=class_token)
    if mask is not None:
        return make_attention_mask(img, mask=mask, patch_size=patch_size, topk_ratio=topk_ratio, class_token=False)
    
    # Original visualization code
    img_array = np.array(img.convert("RGB")) / 255.0
    source = source.detach().cpu()

    h, w, _ = img_array.shape
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

        color = (mask * img_array).sum(axis=(0, 1)) / mask.sum()
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
    
    # img = Image.open("./demo/n01484850_3529.jpeg")
    # img = Image.open("./demo/n02510455_205.jpeg")
    img = Image.open("./demo/n01614925_2611.jpeg")
    vis = transform_vis(img)
    img_tensor = transform_norm(img)[None, ...]
    model.eval()
    
    # Store attention scores if available
    attention_scores = None
    if hasattr(model, 'get_attention_scores'):
        attention_scores = model.get_attention_scores(img_tensor)
    
    _ = model(img_tensor)

    if args.save_vis:
        attn_source = model._tome_info["source"]
        attention_scores = model._tome_info["attn"]
        retain_tokens = int(args.topk_ratio * attention_scores.shape[-1])
        attention_scores = attention_scores.sum(dim=1)
        mask = process_attention_map(attention_scores, retain_tokens, attn_based='tome')
        print(mask.shape)
        if attn_source is not None:
            attention_scores = token_unmerge(mask, attn_source)
            print(attention_scores.sum(-1))
            if attn_source is None:
                raise ValueError("The model does not support ToMe visualization. Please use a model that supports ToMe visualization.")
            print(f"{attn_source.shape[1]} tokens at the end")
            # Create original visualization
            vis_original = make_visualization(vis, attn_source, args.patch_size)
            vis_original.save(osp.join(work_dir, '{}_{}_tokens.png'.format(args.tome, attn_source.shape[1])))
            print("Original visualization saved...")
            
            # Create attention mask visualization if attention scores are available
            if attention_scores is not None:
                vis_attention = make_visualization(
                    vis, attn_source, args.patch_size, 
                    mask=attention_scores[:,1:],
                    topk_ratio=args.topk_ratio
                )
                vis_attention.save(osp.join(work_dir, '{}_{}_tokens_attention.png'.format(args.tome, attn_source.shape[1])))
                print("Attention mask visualization saved...")
        else:
            vis_attention = make_visualization(
                    vis, attn_source, args.patch_size, 
                    attention_scores=attention_scores,
                    topk_ratio=args.topk_ratio,
                    class_token=True
                )
            vis_attention.save(osp.join(work_dir, 'topk_tokens_attention.png'))
            print("Attention mask visualization saved...")


def process_attention_map(attn, top_k, attn_based='topk'):
        batch_size = attn.shape[0]
        flat_attn = attn[:, :, 0] if attn.shape[-1] == 49 else attn[:, 1:, 0]  # 49 For SwinTransformer

        # TODO Version 1.0
        if attn_based == 'topk' or attn_based == 'tome':
            _, indices = torch.topk(flat_attn, top_k, dim=1)
        elif attn_based == 'random':
            random_indices = torch.randperm(flat_attn.size(-1)).expand(batch_size, -1)
            indices = random_indices[:, :top_k]   
        flat_attn_ = torch.zeros_like(flat_attn)
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, top_k)
        flat_attn_[batch_indices, indices] = 1
        return flat_attn_


def parse_args():
    parser = argparse.ArgumentParser(
        description='OpenToMe test (and eval) a model')
    # Basic parameters
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224', help='evaluation model name')
    parser.add_argument('--patch_size', type=int, default=16, help='model patch size')
    parser.add_argument('--tome', type=str, default='none', help='ToMe implementation to use, options: [tome, none]')
    parser.add_argument('--merge_num', type=int, default=98, help='the number of merge tokens')
    parser.add_argument('--merge_ratio', type=float, default=None, help='the ratio of merge tokens in per layers')
    parser.add_argument('--inflect', type=float, default=-0.5, help='the inflect of merge ratio, default: -0.5')
    parser.add_argument('--save_vis', type=bool, default=True, help='whether to save the visualization of the merge tokens')
    # Attention mask parameters
    parser.add_argument('--topk_ratio', type=float, default=0.5, help='ratio of tokens to keep based on attention scores')
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
        raise ValueError('Evaluation with a single process on 1 GPUs.')
    assert args.rank >= 0

    # build the model
    model = timm.create_model(args.model_name, pretrained=True)
    if model == None:
        raise ValueError(f"Model '{args.model_name}' could not be created.")
    if args.local_rank == 0:
        print(f'Model {args.model_name} loaded successfully., param count:{sum([m.numel() for m in model.parameters()])}')

    # Add method to extract attention scores if not already present
    if not hasattr(model, 'get_attention_scores'):
        # This is a placeholder - you'll need to implement this based on your model architecture
        # For many ViT models, you can register hooks to capture attention scores
        def get_attention_scores(self, x):
            # This is a simplified implementation - you'll need to adapt it to your specific model
            # For demonstration purposes, we'll return random scores
            num_tokens = (x.shape[2] // args.patch_size) * (x.shape[3] // args.patch_size) + 1
            return torch.rand(1, 1, num_tokens)  # Random attention scores
            
        model.get_attention_scores = get_attention_scores.__get__(model, type(model))

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
        dtem.dtem_apply_patch(model, feat_dim=None)  # external feature dim, default: none
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