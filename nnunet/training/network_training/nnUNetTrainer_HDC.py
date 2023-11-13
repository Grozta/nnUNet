import os

import numpy as np
import torch
from torch import nn

from typing import Tuple
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.network_architecture.hdc_net_for_unimatch import HDC_Net
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from collections import OrderedDict


class nnUNetTrainer_HDC(nnUNetTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16)

    def process_plans(self, plans):
        super().process_plans(plans)
        self.batch_size = 4
        self.patch_size = [64,160,160]

    def initialize_network(self):
    
        self.max_num_epochs = 400
        self.num_batches_per_epoch = 210   
        self.use_progress_bar = True
        self.feature_channels = [80,96,128,256]

        self.network = HDC_Net(in_chns=1, class_num=14,feature_chns=self.feature_channels)
        
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
        
    def do_split(self):
        """
        This is a suggestion for if your dataset is a dictionary (my personal standard)
        :return:
        """
        dataset_properties = load_pickle(join(self.dataset_directory, 'dataset_properties.pkl'))
        self.patient_identifiers_labeled,self.patient_identifiers_unlabeled = dataset_properties['patient_identifiers_labeled'],dataset_properties['patient_identifiers_unlabeled']
        
        splits_file = join(self.dataset_directory, "splits_final.pkl")
        if not isfile(splits_file):
            self.print_to_log_file("Creating new split...")
            splits = []
            
            all_keys_sorted = np.sort(self.patient_identifiers_labeled)
            kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
            for i, (train_idx, val_idx) in enumerate(kfold.split(all_keys_sorted)):
                train_keys = np.array(all_keys_sorted)[train_idx]
                val_keys = np.array(all_keys_sorted)[val_idx]
                splits.append(OrderedDict())
                splits[-1]['train'] = train_keys
                splits[-1]['val'] = val_keys
            save_pickle(splits, splits_file)

        splits = load_pickle(splits_file)

        if self.fold == "all":
            tr_keys = val_keys = list(self.patient_identifiers_labeled)
        else:
            tr_keys = splits[self.fold]['train']
            val_keys = splits[self.fold]['val']

        tr_keys.sort()
        val_keys.sort()

        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]

        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]
            
        self.dataset_unlabeled = OrderedDict()
        for i in self.patient_identifiers_unlabeled:
            self.dataset_unlabeled[i] = self.dataset[i]
    