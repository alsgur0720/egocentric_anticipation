# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = ['build_feature_head']

import torch
import torch.nn as nn
import torch.nn.functional as F


from src.rekognition_online_action_detection.utils.registry import Registry
from . import transformer as tr
from src.rekognition_online_action_detection.models.transformer.multihead_attention import MultiheadAttentionStream as MultiheadAttention

FEATURE_HEADS = Registry()
FEATURE_SIZES = {
    'rgb_anet_resnet50': 2048,
    'flow_anet_resnet50': 2048,
    'rgb_kinetics_bninception': 1024,
    'audio': 1024,
    'localization':1024,
    'flow_kinetics_bninception': 1024,
    'rgb_kinetics_resnet50': 2048,
    'flow_kinetics_resnet50': 2048,
}

class Adaptiveattention(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(Adaptiveattention, self).__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.tgt_cache = None

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(Adaptiveattention, self).__setstate__(state)

    def stream_inference(self, tgt, memory, pos, tgt_mask=None, memory_mask=None,
                         tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if self.tgt_cache is None:
            tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            self.tgt_cache = tgt
        else:
            tgt = self.tgt_cache
        tgt2 = self.multihead_attn.stream_inference(tgt, memory, memory, pos, attn_mask=memory_mask,
                                                    key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, knn=False):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask, knn=knn)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask, knn=knn)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu

    raise RuntimeError('activation should be relu/gelu, not {}'.format(activation))


@FEATURE_HEADS.register('EK100')
class BaseFeatureHead(nn.Module):

    def __init__(self, cfg):
        super(BaseFeatureHead, self).__init__()

        self.num_heads = cfg.MODEL.LSTR.NUM_HEADS
        self.dim_feedforward = cfg.MODEL.LSTR.DIM_FEEDFORWARD
        self.dropout = cfg.MODEL.LSTR.DROPOUT
        self.activation = cfg.MODEL.LSTR.ACTIVATION
        self.num_classes = cfg.DATA.NUM_CLASSES
        
        self.work_fusions = nn.ModuleList()
        self.fut_fusions = nn.ModuleList()
        
        
        if cfg.INPUT.MODALITY in ['visual', 'motion', 'twostream']:
            self.with_visual = 'motion' not in cfg.INPUT.MODALITY
            self.with_motion = 'visual' not in cfg.INPUT.MODALITY
        else:
            raise RuntimeError('Unknown modality of {}'.format(cfg.INPUT.MODALITY))

        if self.with_visual and self.with_motion:
            visual_size = FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE]
            motion_size = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]
            audio_size = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]
            fusion_size = visual_size + motion_size + audio_size
        elif self.with_visual:
            visual_size = FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE]
        elif self.with_motion:
            fusion_size = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]

        # self.d_model = visual_size
        self.d_model = fusion_size
        self.work_adapt_layer = Adaptiveattention(
                            2048, self.num_heads, self.dim_feedforward,
                            self.dropout, self.activation)
        
        
        self.work_adapt_layer1 = Adaptiveattention(
                            2048, self.num_heads, self.dim_feedforward,
                            self.dropout, self.activation)
        
        self.work_adapt_layer2 = Adaptiveattention(
                            2048, self.num_heads, self.dim_feedforward,
                            self.dropout, self.activation)
        
        
        
        
        if cfg.MODEL.FEATURE_HEAD.LINEAR_ENABLED:
            if cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES != -1:
                self.d_model = cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES
            if self.with_motion:
                self.motion_linear = nn.Sequential(
                    nn.Linear(motion_size, self.d_model),
                    nn.LayerNorm(self.d_model),
                    nn.ReLU(inplace=True)
                )
            if self.with_visual:
                self.visual_linear = nn.Sequential(
                    nn.Linear(visual_size, self.d_model),
                    nn.LayerNorm(self.d_model),
                    nn.ReLU(inplace=True)
                )
            
            if self.with_motion and self.with_visual:
                self.audio_linear = nn.Sequential(
                    nn.Linear(visual_size, self.d_model),
                    nn.LayerNorm(self.d_model),
                    nn.ReLU(inplace=True)
                )
                self.input_linear = nn.Sequential(
                    # nn.Linear(2 * self.d_model, self.d_model),
                    nn.Linear(self.d_model, self.d_model),
                    nn.LayerNorm(self.d_model),
                    nn.ReLU(inplace=True)
                )
        else:
            if self.with_motion:
                self.motion_linear = nn.Identity()
            if self.with_visual:
                self.visual_linear = nn.Identity()
            if self.with_motion and self.with_visual:
                self.input_linear = nn.Identity()

    def forward(self, visual_input, motion_input, audio_input):
        
        visual_input = self.visual_linear(visual_input)
        motion_input  = self.motion_linear(motion_input)
        audio_input  = self.audio_linear(audio_input)
        
        visual_input = self.work_adapt_layer(visual_input, visual_input)
            
        audio_input = self.work_adapt_layer1(visual_input, audio_input)
            
        motion_input = self.work_adapt_layer2(audio_input, motion_input)
        fusion_input = self.input_linear(motion_input)

        return fusion_input


def build_feature_head(cfg):
    feature_head = FEATURE_HEADS[cfg.DATA.DATA_NAME]
    return feature_head(cfg)
