import os
import cv2
import glob
import logging
import argparse

import torch
import torch.utils.data

from fire import Fire
from tqdm import tqdm
from moviepy.editor import VideoFileClip

from ssd.config import cfg
from ssd.utils import dist_util
from ssd.utils.logger import setup_logger
from ssd.utils.dist_util import synchronize
from ssd.utils.checkpoint import CheckPointer
from ssd.engine.inference import do_evaluation
from ssd.modeling.detector import build_detection_model

def main(video, config):
    class_name = ('__background__',
            'lubang', 'retak aligator', 'retak melintang', 'retak memanjang')

    cfg.merge_from_file(config)
    cfg.freeze()

    ckpt = None
    device = torch.device(cfg.MODEL.DEVICE)
    model = build_detection_model(cfg)
    model.to(device)

    checkpoint = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpoint.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpoint.get_checkpoint_file()
    print(f'Loading weight from {weight_file}')

def evaluation(cfg, ckpt, distributed):
    logger = logging.getLogger('SSD.inference')

    model = build_detection_model(cfg)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR, logger=logger)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    do_evaluation(cfg, model, distributed)

if __name__ == '__main__':
    Fire(main)
