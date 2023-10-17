import os

import numpy as np
import torch
from torch import nn

from typing import Tuple
from nnunet.network_architecture.hdc_net import HDC_Net
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper


class nnUNetTrainer_FLARE_HDC(nnUNetTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16)


    def initialize_network(self):
        
        self.folder_with_preprocessed_data = os.path.join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
        if self.threeD:
            conv_op = nn.Conv3d
        else:
            conv_op = nn.Conv2d

        self.max_num_epochs = 500
        self.num_batches_per_epoch = 440    
        self.use_progress_bar = True

        self.network = HDC_Net(in_dim=1, out_dim=14,layer_channels=[128,256,256,384])
        self.network.cuda()
        self.network.conv_op = conv_op
        self.network.num_classes = self.num_classes
        self.network.input_channels_pbl = self.num_input_channels

        if torch.cuda.is_available():
             self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
        
    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["do_elastic"] = True
        
    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision: bool = True) -> Tuple[np.ndarray, np.ndarray]:

        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']

        if do_mirroring:
            assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"

        current_mode = self.network.training
        self.network.eval()
        ret = self.network.predict_3D(data, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                      use_sliding_window=use_sliding_window, step_size=step_size,
                                      patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                                      use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                      pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                      mixed_precision=mixed_precision)
        self.network.train(current_mode)
        return ret
        
    