import os

import numpy as np
import torch
from torch import nn

from typing import Tuple
from nnunet.network_architecture.hdc_net_for_unimatch import HDC_Net
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper


class nnUNetTrainer_HDC(nnUNetTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16)

    def process_plans(self, plans):
        super().process_plans(plans)
        self.batch_size = 4
        self.patch_size = [64,160,160]

    def initialize_network(self):
    
        self.max_num_epochs = 500
        self.num_batches_per_epoch = 120    
        self.use_progress_bar = True

        self.network = HDC_Net(in_chns=1, class_num=14,feature_chns=[64,64,128,128])
        
        if self.threeD:
            self.network.conv_op = nn.Conv3d
            self.network.dropout_op = nn.Dropout3d
            self.network.norm_op = nn.InstanceNorm3d
        else:
            self.network.conv_op = nn.Conv2d
            self.network.dropout_op = nn.Dropout2d
            self.network.norm_op = nn.InstanceNorm2d
            
        self.network.num_classes = self.num_classes
        self.network.input_channels_pbl = self.num_input_channels
        
        self.network.inference_apply_nonlin = softmax_helper
        if torch.cuda.is_available():
             self.network.cuda()
        
    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["do_elastic"] = True
    