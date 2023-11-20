import os

import numpy as np
import torch
from torch import nn
from time import time
from datetime import datetime

from typing import Tuple
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.network_architecture.hdc_net_for_unimatch import HDC_Net
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from collections import OrderedDict
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss_with_weight
from nnunet.training.dataloading.dataset_loading import DataLoader3D_with_selected_wieght
from torch.optim import lr_scheduler
class nnUNetTrainer_HDC(nnUNetTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16)
        dataset_properties = load_pickle(join(self.dataset_directory, 'dataset_properties.pkl'))
        self.dice_weight=dataset_properties["intensityproperties"][0]['Statistics_of_the_number_of_voxels_in_each_organ']
        organ_vol = np.array(self.dice_weight)[1:]
        self.dataset_need_focal_class_weight = (organ_vol.sum()/organ_vol).tolist()   
        v = np.array(self.dice_weight)   
        self.focal_alpha = (v.sum()/v).tolist()
        #self.focal_alpha = self.dice_weight
        self.oversample_foreground_percent = 0.33
        self.loss = DC_and_CE_loss_with_weight({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False,'dice_weight':self.dice_weight}, {'alpha':self.focal_alpha})
        
        self.lr_scheduler_eps = 1e-4
        self.lr_scheduler_patience = 20
        self.initial_lr = 1e-3
        self.lr_scheduler_factor=0.8

    def process_plans(self, plans):
        super().process_plans(plans)
        self.batch_size = 4
        self.patch_size = [64,160,160]  

    def initialize_network(self):
    
        self.max_num_epochs = 600
        self.num_batches_per_epoch = 320   
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
            
    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()
        timestamp = datetime.now()
        if self.threeD:
            dl_tr = DataLoader3D_with_selected_wieght(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,self.dataset_need_focal_class_weight,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_tr.dataset_debug_log = join(self.output_folder, "dataset_debug_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                 (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                                  timestamp.second))
            dl_val = DataLoader3D_with_selected_wieght(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,self.dataset_need_focal_class_weight, 
                                  False,  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val.dataset_debug_log = join(self.output_folder, "dataset_debug_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                 (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                                  timestamp.second))

        return dl_tr, dl_val

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.Adam(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                          amsgrad=True)
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=self.lr_scheduler_factor,
                                                           patience=self.lr_scheduler_patience,
                                                           verbose=True, threshold=self.lr_scheduler_eps,
                                                           threshold_mode="abs")