# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
sys.path.append("")
from src.rekognition_online_action_detection.utils.parser import load_cfg
from src.rekognition_online_action_detection.utils.env import setup_environment
from src.rekognition_online_action_detection.utils.checkpointer import setup_checkpointer, setup_checkpointer_rev
from src.rekognition_online_action_detection.utils.logger import setup_logger
from src.rekognition_online_action_detection.models import build_model, build_rev_model
from src.rekognition_online_action_detection.engines import do_inference_rev


def main(cfg):
    # Setup configurations
    device = setup_environment(cfg)
    checkpointer = setup_checkpointer(cfg, phase='test')
    checkpointer_rev = setup_checkpointer_rev(cfg, phase='test')

    logger = setup_logger(cfg, phase='test')

    # Build model
    model = build_model(cfg, device)
    rev_model = build_rev_model(cfg, device)

    # Load pretrained model
    checkpointer.load(model)
    checkpointer_rev.load(rev_model)

    do_inference_rev(
        cfg,
        model,
        device,
        logger,
    )


if __name__ == '__main__':
    main(load_cfg())
