import os
import os.path as osp
import time
import timm
import torch
from tqdm import tqdm
from opentome.timm import tome, dtem, diffrate, tofu, mctf, crossget, dct, pitome
from opentome.tome import tome as tm
from opentome.utils.datasets import dataset_loader, accuracy
import argparse
import logging

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from opentome.utils import tome_visulization


def demo_eval(model, work_dir, args):
    input_size = model.default_cfg["input_size"][1]

    # Make sure the transform is correct for your model!
    transform_list = [
        transforms.Resize(int((256 / 224) * input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size)
    ]
    # The visualization and model need different transforms
    transform_vis  = transforms.Compose(transform_list)
    transform_norm = transforms.Compose(transform_list + [
        transforms.ToTensor(),
        transforms.Normalize(model.default_cfg["mean"], model.default_cfg["std"]),
    ])
    
    img = Image.open("./demo/n02510455_205.jpeg")
    vis = transform_vis(img)
    img = transform_norm(img)[None, ...]
    model.eval()
    outputs = model(img)
    print(outputs.topk(5).indices[0].tolist())

    if args.save_vis:
        attn_source = model._tome_info["source"]
        if attn_source is None:
            raise ValueError("The model does not support ToMe visualization. Please use a model that supports ToMe visualization.")
        print(f"{attn_source.shape[1]} tokens at the end")
        vis = tome_visulization.make_visualization(vis, attn_source, args.patch_size)
        vis.save(osp.join(work_dir, '{}_{}_vis.png'.format(args.tome, attn_source.shape[1])))
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
    # Dataset parameters
    parser.add_argument('--input_size', type=int, default=None, help='the input resolution')
    parser.add_argument('--dataset', type=str, default='data/ImageNet/val', help='the dataset to use for evaluation')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    # Environment parameters
    parser.add_argument('--work_dir', type=str, default='work_dirs/in1k_classification', help='the dir to save logs and models')
    parser.add_argument('--gpu-id', type=int, default=0, help='id of gpu to use ' '(only applicable to non-distributed testing)')
    parser.add_argument('--launcher', choices=['none', 'slurm', 'pytorch'], default='none', help='job launcher')
    parser.add_argument('--local_rank', help='set local_rank for torch.distributed.launch (torch<2.0.0)', type=int, default=0)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--port', type=int, default=29500, help='port only works when launcher=="slurm"')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        work_dir = osp.join(args.work_dir, 'eval_{}_by_{}'.format(args.model_name, args.tome))
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(work_dir, 'merge_token_{}_{}.log'.format(args.merge_num, timestamp))
    mkdir = osp.dirname(log_file)
    os.makedirs(mkdir, exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger()
    logger.info('Start evaluation with args: {}'.format(args))

    # build the model
    model = timm.create_model(args.model_name, pretrained=True)
    if model == None:
        logger.info(f"Error: Model '{args.model_name}' could not be created.")
        raise ValueError(f"Model '{args.model_name}' could not be created.")
    logger.info(f"Model {args.model_name} loaded successfully.")

    # build the dataloader  -->  /path/imagenet/ 
    #                       --> e.g. yuchang/lsy/.cache/imagenet/val
    if not osp.exists(args.dataset):
        logger.info(f"Error: Dataset path '{args.dataset}' does not exist.")
        raise FileNotFoundError(f"Dataset path '{args.dataset}' does not exist.")
    val_loader = dataset_loader.create_dataset(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=model.default_cfg["input_size"][1] if args.input_size is None else args.input_size,
        mean=model.default_cfg["mean"],
        std=model.default_cfg["std"]
    )
    # val_loader = None

    assert args.merge_num >= 0, "Please specify a positive merge number."
    assert args.inflect in [-0.5, 1, 2], "Please specify a valid inflect value."
    if args.tome in ['tome', 'tofu', 'crossget', 'dct', "pitome"]:
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
            args.merge_num = sum(tm.parse_r(len(model.blocks), r=(args.merge_ratio, args.inflect)))
        elif args.merge_ratio is None and args.merge_num is not None:
            merge_ratio = tm.check_parse_r(len(model.blocks), args.merge_num, 
                                    (model.default_cfg["input_size"][1]/args.patch_size) ** 2, args.inflect)
        # update _tome_info
        model.r = (merge_ratio, args.inflect)
        model._tome_info["r"] = model.r
        model._tome_info["total_merge"] = args.merge_num
        logger.info(model._tome_info)
    elif args.tome == 'dtem':
        dtem.dtem_apply_patch(model, feat_dim=None)  # exteranal feature dim, defalut: none
        if not hasattr(model, '_tome_info'):
            raise ValueError("The model does not support DTEM. Please use a model that supports DTEM.")
        if args.merge_ratio is not None and args.merge_num is None:
            args.merge_num = sum(tm.parse_r(len(model.blocks), r=(args.merge_ratio, args.inflect)))
        elif args.merge_ratio is None and args.merge_num is not None:
            merge_ratio = tm.check_parse_r(len(model.blocks), args.merge_num, 
                                    (model.default_cfg["input_size"][1]/args.patch_size) ** 2, args.inflect)
        # update _tome_info
        model.r = (merge_ratio, args.inflect)
        model._tome_info["r"] = model.r
        model._tome_info["k2"] = 3
        model._tome_info["tau1"] = 0.1
        model._tome_info["tau2"] = 0.1
        model._tome_info["total_merge"] = args.merge_num
        logger.info(model._tome_info)
    elif args.tome == 'diffrate':
        diffrate.diffrate_apply_patch(model, prune_granularity=4, merge_granularity=4)
        if not hasattr(model, '_tome_info'):
            raise ValueError("The model does not support DiffRate. Please use a model that supports DiffRate.")
        r = args.merge_num / len(model.blocks) if args.merge_num is not None else 0
        model.init_kept_num_using_r(int(r))
        logger.info(model._tome_info)
    elif args.tome == 'mctf':
        mctf.mctf_apply_patch(model)
        if not hasattr(model, '_tome_info'):
            raise ValueError("The model does not support MCTF. Please use a model that supports MCTF.")
        if args.merge_ratio is not None and args.merge_num is None:
            args.merge_num = sum(tm.parse_r(len(model.blocks), r=(args.merge_ratio, args.inflect)))
        elif args.merge_ratio is None and args.merge_num is not None:
            merge_ratio = tm.check_parse_r(len(model.blocks), args.merge_num, 
                                    (model.default_cfg["input_size"][1]/args.patch_size) ** 2, args.inflect)
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
        logger.info(model._tome_info)
    elif args.tome == 'none':
        pass
    else:
        raise ValueError("Invalid ToMe implementation specified. Use 'tome' or 'none'.")

    # For debugging...
    demo_eval(model, work_dir, args)

    # evaluate the model
    total_top1, total_top5 = 0, 0
    total_samples = 0
    model.eval()
    if torch.cuda.is_available():
        model.cuda(args.gpu_id)
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images, labels = images.cuda(), labels.cuda() if torch.cuda.is_available() else (images, labels)
            outputs = model(images)
            results = accuracy.accuracy_one_hot(outputs, labels, (1, 5))

            total_top1 += results[0].item() * args.batch_size
            total_top5 += results[-1].item() * args.batch_size
            total_samples += args.batch_size
        logger.info(f"Final accuracy: Top-1: {total_top1 / total_samples:.2f}%, Top-5: {total_top5 / total_samples:.2f}%")


if __name__=="__main__":
    main()

