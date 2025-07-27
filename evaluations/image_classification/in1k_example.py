import re
import os
import os.path as osp
import time
import timm
import torch
import torch.distributed as dist
from timm.utils import AverageMeter
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from opentome.timm import tome, dtem, diffrate, tofu, mctf, crossget, dct, pitome
from opentome.tome import tome as tm
from opentome.utils.datasets import dataset_loader, accuracy
import argparse
import logging

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(
        description='OpenToMe test (and eval) a model')
    # Baiscal parameters
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224', help='evaluation model name')
    parser.add_argument('--patch_size', type=int, default=16, help='model patch size')
    parser.add_argument('--tome', type=str, default='none', help='ToMe implementation to use, options: [tome, none]')
    parser.add_argument('--merge_num', type=str, default='98', help='the number of merge tokens')
    parser.add_argument('--merge_ratio', type=float, default=None, help='the ratio of merge tokens in per layers')
    parser.add_argument('--inflect', type=float, default=-0.5, help='the inflect of merge ratio, default: -0.5')
    parser.add_argument('--save_vis', type=bool, default=True, help='whether to save the visualization of the merge tokens')
    # Dataset parameters
    parser.add_argument('--input_size', type=int, default=None, help='the input resolution')
    parser.add_argument('--dataset', type=str, default='data/ImageNet/val', help='the dataset to use for evaluation')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loading')
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

def cleanup():
    dist.destroy_process_group()


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt *= world_size
    return rt


def evaluation(args):

    args = parse_args()

    # Split String to list.
    items = re.split(r'[_\-,.\s]+', args.merge_num)
    merge_list = [int(item) for item in items if item.strip()] 

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        work_dir = osp.join(args.work_dir, 'eval_{}_by_{}'.format(args.model_name, args.tome))
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(work_dir, 'cls_{}.log'.format(timestamp))
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
        logger.info('Evaulation in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        logger.info('Evaulation with a single process on 1 GPUs.')
    assert args.rank >= 0

    # build the model
    model = timm.create_model(args.model_name, pretrained=True)
    if model == None:
        logger.info(f"Error: Model '{args.model_name}' could not be created.")
        raise ValueError(f"Model '{args.model_name}' could not be created.")
    if args.local_rank == 0:
        logger.info(
            f'Model {args.model_name} loaded successfully., param count:{sum([m.numel() for m in model.parameters()])}')

    # setup distributed training
    model = model.to(args.device)
    if args.distributed:
        if args.local_rank == 0:
            logger.info("Using native Torch DistributedDataParallel.")
        model = DDP(model, device_ids=[args.local_rank])

    if not osp.exists(args.dataset):
        logger.info(f"Error: Dataset path '{args.dataset}' does not exist.")
        raise FileNotFoundError(f"Dataset path '{args.dataset}' does not exist.")
    dataset = dataset_loader.create_dataset(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=model.module.default_cfg["input_size"][1] if args.input_size is None else args.input_size,
        mean=model.module.default_cfg["mean"],
        std=model.module.default_cfg["std"],
        return_dataset_only=True
    )
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=args.world_size,
        rank=args.rank
    )
    # DataLoader
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
    )

    # Evaluation
    for merge_num in merge_list:
        assert merge_num >= 0, "Please specify a positive merge number."
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
            if args.merge_ratio is not None and merge_num is None:
                merge_num = sum(tm.parse_r(len(model.module.blocks), r=(args.merge_ratio, args.inflect)))
            elif args.merge_ratio is None and merge_num is not None:
                merge_ratio = tm.check_parse_r(len(model.module.blocks), merge_num, 
                                        (model.module.default_cfg["input_size"][1] / args.patch_size) ** 2, args.inflect)
            # update _tome_info
            model.r = (merge_ratio, args.inflect)
            model._tome_info["r"] = model.r
            model._tome_info["total_merge"] = merge_num
            logger.info(model._tome_info)
        elif args.tome.lower() == 'dtem':
            dtem.dtem_apply_patch(model, feat_dim=None)  # exteranal feature dim, defalut: none
            if not hasattr(model, '_tome_info'):
                raise ValueError("The model does not support DTEM. Please use a model that supports DTEM.")
            if args.merge_ratio is not None and merge_num is None:
                merge_num = sum(tm.parse_r(len(model.module.blocks), r=(args.merge_ratio, args.inflect)))
            elif args.merge_ratio is None and merge_num is not None:
                merge_ratio = tm.check_parse_r(len(model.module.blocks), merge_num, 
                                        (model.module.default_cfg["input_size"][1]/args.patch_size) ** 2, args.inflect)
            # update _tome_info
            model.r = (merge_ratio, args.inflect)
            model._tome_info["r"] = model.r
            model._tome_info["k2"] = 3
            model._tome_info["tau1"] = 0.1
            model._tome_info["tau2"] = 0.1
            model._tome_info["total_merge"] = merge_num
            logger.info(model._tome_info)
        elif args.tome.lower() == 'diffrate':
            diffrate.diffrate_apply_patch(model, prune_granularity=4, merge_granularity=4)
            if not hasattr(model, '_tome_info'):
                raise ValueError("The model does not support DiffRate. Please use a model that supports DiffRate.")
            r = merge_num / len(model.module.blocks) if merge_num is not None else 0
            model.init_kept_num_using_r(int(r))
            logger.info(model._tome_info)
        elif args.tome.lower() == 'mctf':
            mctf.mctf_apply_patch(model)
            if not hasattr(model, '_tome_info'):
                raise ValueError("The model does not support MCTF. Please use a model that supports MCTF.")
            if args.merge_ratio is not None and merge_num is None:
                merge_num = sum(tm.parse_r(len(model.module.blocks), r=(args.merge_ratio, args.inflect)))
            elif args.merge_ratio is None and merge_num is not None:
                merge_ratio = tm.check_parse_r(len(model.module.blocks), merge_num, 
                                        (model.module.default_cfg["input_size"][1]/args.patch_size) ** 2, args.inflect)
            # update _tome_info
            model.r = (merge_ratio, args.inflect)
            model._tome_info["r"] = model.r
            model._tome_info["total_merge"] = merge_num
            model._tome_info["one_step_ahead"] = 1
            model._tome_info["tau_sim"] = 1
            model._tome_info["tau_info"] = 20
            model._tome_info["tau_size"] = 40
            model._tome_info["bidirection"] = True
            model._tome_info["pooling_type"] = 'none'
            logger.info(model._tome_info)
        elif args.tome.lower() == 'none':
            pass
        else:
            raise ValueError("Invalid ToMe implementation specified. Use 'tome' or 'none'.")

        # evaluate the model
        top1 = AverageMeter()
        top5 = AverageMeter()

        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Evaluating"):
                images, labels = images.to(args.device), labels.to(args.device)
                outputs = model(images)
                results = accuracy.accuracy_one_hot(outputs, labels, (1, 5))

                if args.distributed:
                    acc1 = reduce_tensor(results[0], args.world_size)
                    acc5 = reduce_tensor(results[-1], args.world_size)
                torch.cuda.synchronize()

                top1.update(acc1.item(), outputs.size(0))
                top5.update(acc5.item(), outputs.size(0))
            
            if args.rank == 0:
                logger.info(f"Final accuracy: Top-1: {top1.avg}%, Top-5: {top5.avg}%")

    cleanup()


def main():
    args = parse_args()
    evaluation(args)


if __name__=="__main__":
    main()

