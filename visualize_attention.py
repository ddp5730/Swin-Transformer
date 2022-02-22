# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from config import get_config
from data import build_loader
from logger import create_logger
from models import build_model
from utils import replace_fc_layer

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--transfer-dataset', action='store_true', help='Transfer the model to a new dataset')
    # TODO: See if TransFG modifies model in any other way when performing transfer
    # Doesn't look like it, but I'm not ready to rule it out yet.

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    # config.DATA.BATCH_SIZE = 1  # Set batch size to 1 for ease of use
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    if config.TRAIN.TRANSFER_DATASET:
        replace_fc_layer(config, model)
        model.cuda()

    if config.MODEL.RESUME:
        load_checkpoint(config, model, logger)
        if config.EVAL_MODE:
            return

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    # Visualize ten attention maps from each class
    visualize_attention_maps(data_loader_val, model, config)


def load_checkpoint(config, model, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)

    del checkpoint
    torch.cuda.empty_cache()
    return


def visualize_attention_maps(data_loader, model, config):
    """
    Method to visualize ten random attention maps for each class.  Outputs
    are saved to the output directory
    """
    model.eval()

    class_images = torch.zeros((config.MODEL.NUM_CLASSES, 10, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                               dtype=torch.float32)
    classes = data_loader.dataset.classes.copy()
    num_images = np.zeros((len(classes)), dtype=np.uint8)

    for data, label in data_loader:
        data = data.squeeze()
        # Data is of type torch.float32
        test = str(label.squeeze().item()+1)
        class_idx = classes.index(test)

        num_class_images = num_images[class_idx]
        if num_class_images < 10:
            class_images[class_idx, num_class_images, :, :, :] = data
            num_images[class_idx] += 1

        # Check if all classes have 10 images
        needs_more_images = False
        for i in range(num_images.size):
            num_class_images = num_images[i]
            if num_class_images < 10:
                needs_more_images = True
                break

        if not needs_more_images:
            break

    class_images = class_images.cuda()
    for class_num in range(config.MODEL.NUM_CLASSES):
        imgs = class_images[class_num, :, :, :, :]
        logits = model(imgs)
        test = 1 + 1

    return


if __name__ == '__main__':
    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
