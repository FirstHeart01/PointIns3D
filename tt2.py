import functools

import numpy as np
import torch
import torch.onnx
import yaml
from munch import Munch
from torch import nn
import os
import os.path as osp
import shutil
import time

import spconv.pytorch as spconv

from pointins3d.ops import (ball_query, bfs_cluster, get_mask_iou_on_cluster, get_mask_iou_on_pred,
                            get_mask_label, global_avg_pool, sec_max, sec_min, voxelization,
                            voxelization_idx)
from pointins3d.data import build_dataset, build_dataloader
from pointins3d.model import PointIns3D
from pointins3d.util import (AverageMeter, SummaryWriter, build_optimizer, checkpoint_save,
                             collect_results_cpu, cosine_lr_after_step, get_dist_info,
                             get_max_memory, get_root_logger, init_dist, is_main_process,
                             is_multiple, is_power2, load_checkpoint)
from pointins3d.model.blocks import ResidualBlock, UBlock
from torchstat import stat

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    cfg_path = 'configs/pointins3d_stpls3d_backbone_res16_msc_trans.yaml'
    cfg_txt = open(cfg_path, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    # work_dir & logger
    cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(cfg_path))[0])
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file)
    logger.info(f'Config:\n{cfg_txt}')
    logger.info(f'Mix precision training: {cfg.fp16}')
    shutil.copy(cfg_path, osp.join(cfg.work_dir, osp.basename(cfg_path)))
    writer = SummaryWriter(cfg.work_dir)
    # model
    model = PointIns3D(**cfg.model).cuda()
    # data
    
    train_set = build_dataset(cfg.data.train, logger=logger)
    count = dict()
    for i, batch in enumerate(train_set):
        semantic_labels = batch[4]
        for i in range(cfg.model.semantic_classes): # 0代表ground，会忽略
            if i not in count:
                count[i] = torch.sum(semantic_labels==i)
            else:
                count[i] += torch.sum(semantic_labels==i)
    logger.info('count:{}'.format(count))
    


if __name__ == '__main__':
    main()
