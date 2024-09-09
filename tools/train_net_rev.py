# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
sys.path.append("")
from src.rekognition_online_action_detection.utils.parser import load_cfg
from src.rekognition_online_action_detection.utils.env import setup_environment
from src.rekognition_online_action_detection.utils.checkpointer import setup_checkpointer, setup_checkpointer_rev
from src.rekognition_online_action_detection.utils.logger import setup_logger
from src.rekognition_online_action_detection.datasets import build_data_loader
from src.rekognition_online_action_detection.models import build_model, build_rev_model
from src.rekognition_online_action_detection.criterions import build_criterion
from src.rekognition_online_action_detection.optimizers import build_optimizer
from src.rekognition_online_action_detection.optimizers import build_scheduler
from src.rekognition_online_action_detection.optimizers import build_ema
from src.rekognition_online_action_detection.engines import do_train_for_rev


def main(cfg):
    # Setup configurations
    device = setup_environment(cfg)
    checkpointer = setup_checkpointer(cfg, phase='test')
    checkpointer2 = setup_checkpointer_rev(cfg, phase='train')
    logger = setup_logger(cfg, phase='train')

    # Build data loaders
    data_loaders = {
        phase: build_data_loader(cfg, phase)
        for phase in cfg.SOLVER.PHASES
    }

    # Build model
    model = build_model(cfg, device)
    rev_model = build_rev_model(cfg, device)

    # Build criterion
    criterion = build_criterion(cfg, device)

    # Build optimizer
    optimizer = build_optimizer(cfg, model)

    # Build ema
    ema = build_ema(model, 0.999)

    # Load pretrained model and optimizer
    checkpointer.load(model, optimizer)
    checkpointer2.load(rev_model, optimizer)

    # Build scheduler
    scheduler = build_scheduler(
        cfg, optimizer, len(data_loaders['train']))

    do_train_for_rev(
        cfg,
        data_loaders,
        model, rev_model,
        criterion,
        optimizer,
        scheduler,
        ema,
        device,
        checkpointer2,
        logger,
    )


if __name__ == '__main__':
    main(load_cfg())
