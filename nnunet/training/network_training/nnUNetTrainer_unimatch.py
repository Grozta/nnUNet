import os
from _warnings import warn
from time import time, sleep
from torch.cuda.amp import GradScaler, autocast

import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from tqdm import trange
from batchgenerators.utilities.file_and_folder_operations import *
from sklearn.model_selection import KFold
from collections import OrderedDict

from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D, unpack_dataset,DataLoader3D_for_unimatch
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params, \
    default_2D_augmentation_params, get_semi_augmentation, get_patch_size

from typing import Tuple
from nnunet.network_architecture.hdc_net_for_unimatch import HDC_Net
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda, maybe_to_cuda

from nnunet.training.loss_functions.dice_loss import DiceLoss

class nnUNetTrainer_unimatch(nnUNetTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16)
                
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_dice = DiceLoss()
        self.conf_thresh = 0.95
    
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
        
    def get_semi_generators(self):
        self.load_dataset()
        
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_unlabeled = DataLoader3D_for_unimatch(self.dataset_unlabeled, self.basic_generator_patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        else:
            dl_tr = DataLoader2D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_unlabeled = DataLoader3D_for_unimatch(self.dataset_unlabeled, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        return dl_tr, dl_val, dl_unlabeled
    
    def initialize(self, training=True, force_load_plans=False):
        """
        For prediction of test cases just set training=False, this will prevent loading of training data and
        training batchgenerator initialization
        :param training:
        :return:
        """

        maybe_mkdir_p(self.output_folder)

        if force_load_plans or (self.plans is None):
            self.load_plans_file()

        self.process_plans(self.plans)

        self.setup_DA_params()
        
        self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                            "_stage%d" % self.stage)

        if training:


            self.dl_tr, self.dl_val, self.dl_unlabeled = self.get_semi_generators()
            if self.unpack_data:
                self.print_to_log_file("unpacking dataset")
                unpack_dataset(self.folder_with_preprocessed_data)
                self.print_to_log_file("done")
            else:
                self.print_to_log_file(
                    "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                    "will wait all winter for your model to finish!")
            self.tr_gen, self.val_gen, self.unlabeled_gen = get_semi_augmentation(self.dl_tr, self.dl_val, self.dl_unlabeled,
                                                                 self.data_aug_params['patch_size_for_spatialtransform'],
                                                                 self.data_aug_params)
            self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                   also_print_to_console=False)
            self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                   also_print_to_console=False)
            self.print_to_log_file("UNLABELED KEYS:\n %s" % (str(self.dataset_val.keys())),
                                   also_print_to_console=False)
        else:
            pass
        self.initialize_network()
        self.initialize_optimizer_and_scheduler()
        # assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        self.was_initialized = True
    
    def process_plans(self, plans):
        super().process_plans(plans)
        self.batch_size = 1
        self.basic_generator_patch_size= [80,188,188] # gl_gen第一次加载数据时patch化的大小
        self.patch_size = [64,160,160]  # 数据最后加工完毕之后输出的大小

    def initialize_network(self):
        
        self.max_num_epochs = 100
        self.num_batches_per_epoch = 600    
        self.use_progress_bar = True
        
        self.network = HDC_Net(in_chns=1, class_num=14,feature_chns=[64,96,128,128])
        
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
        # do self DA
        
    def run_training(self):
        self.save_debug_information()
        
        #super(nnUNetTrainer, self).run_training()
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        _ = self.tr_gen.next()
        _ = self.unlabeled_gen.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)        
        # self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            # train one epoch
            self.network.train()

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))

                        l = self.run_iteration_unimatch(self.tr_gen, self.unlabeled_gen, True)

                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
            else:
                for _ in range(self.num_batches_per_epoch):
                    l = self.run_iteration_unimatch(self.tr_gen, self.unlabeled_gen ,True)
                    train_losses_epoch.append(l)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, True)
                    val_losses.append(l)
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration_unimatch(self.val_gen, self.unlabeled_gen , False)
                        val_losses.append(l)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))
            
    def run_iteration_unimatch(self, labeled_data_generator, unlabeled_data_generator, do_backprop=True, run_online_evaluation=False):
        labeled_data_dict = next(labeled_data_generator)
        unlabeled_data_dict = next(unlabeled_data_generator)
        """        
        return {'image_u_w': unlabeled_1, 'image_u_w_mix': unlabeled_2,'image_u_s1': None,'image_u_s2': None,
        "mixcut_box_1": None,"mixcut_box_2":None,"image_identifier":image_identifier}
        """
        img_x = labeled_data_dict['data']
        mask_x = labeled_data_dict['target']
        mask_x = mask_x.squeeze(1)
        img_u_w = unlabeled_data_dict['image_u_w']
        img_u_w_mix = unlabeled_data_dict['image_u_w_mix']
        img_u_s1 = unlabeled_data_dict['image_u_s1']
        img_u_s2 = unlabeled_data_dict['image_u_s2']
        cutmix_box1 = unlabeled_data_dict['mixcut_box_1']
        cutmix_box2 = unlabeled_data_dict['mixcut_box_2']

        if self.fp16:
            with autocast():
                with torch.no_grad():
                    self.network.eval()
                    img_u_w_mix = maybe_to_cuda(img_u_w_mix)
                    pred_u_w_mix = self.network(img_u_w_mix).detach().cpu().float()
                    conf_u_w_mix = (pred_u_w_mix.softmax(dim=1).max(dim=1)[0]).float()
                    mask_u_w_mix = (pred_u_w_mix.argmax(dim=1))
                
                    del img_u_w_mix
                    torch.cuda.empty_cache()
                    
                self.network.train()
                self.optimizer.zero_grad()
                
                num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
                img_x, img_u_w, img_u_s1, img_u_s2 = maybe_to_cuda([img_x,img_u_w,img_u_s1,img_u_s2])
                preds, preds_fp = self.network(torch.cat((img_x, img_u_w)), True)
                pred_x, pred_u_w = preds.split([num_lb, num_ulb])
                pred_u_w_fp = preds_fp[num_lb:]
                 
                pred_u_s1, pred_u_s2 = self.network(torch.cat((img_u_s1, img_u_s2))).chunk(2)
                pred_u_w = pred_u_w.detach().cpu().float()
                conf_u_w = (pred_u_w.softmax(dim=1).max(dim=1)[0]).float()
                mask_u_w = pred_u_w.argmax(dim=1)
                
                mask_u_w_cutmixed1, conf_u_w_cutmixed1 = mask_u_w.clone(), conf_u_w.clone()
                mask_u_w_cutmixed2, conf_u_w_cutmixed2 = mask_u_w.clone(), conf_u_w.clone()
                
                mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
                conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]

                mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
                conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
                
                mask_x = mask_x.cuda().long()
                mask_u_w_cutmixed1 = mask_u_w_cutmixed1.cuda().float()
                mask_u_w_cutmixed2 = mask_u_w_cutmixed2.cuda().float()
                mask_u_w = mask_u_w.cuda()
                
                ce_loss = self.criterion_ce(pred_x.float(), mask_x) + 1e-8
                dice_loss = self.criterion_dice(pred_x.softmax(dim=1), mask_x.unsqueeze(1))

                loss_x = (ce_loss+dice_loss)/ 2.0

                loss_u_s1 = self.criterion_dice(pred_u_s1.softmax(dim=1), mask_u_w_cutmixed1.unsqueeze(1),
                                        ignore=(conf_u_w_cutmixed1 < self.conf_thresh))
                
                loss_u_s2 = self.criterion_dice(pred_u_s2.softmax(dim=1), mask_u_w_cutmixed2.unsqueeze(1),
                                        ignore=(conf_u_w_cutmixed2 < self.conf_thresh))
                
                loss_u_w_fp = self.criterion_dice(pred_u_w_fp.softmax(dim=1), mask_u_w.unsqueeze(1),
                                            ignore=(conf_u_w < self.conf_thresh))
                
                l = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            with torch.no_grad():
                    self.network.eval()
                    img_u_w_mix = maybe_to_cuda(img_u_w_mix)
                    pred_u_w_mix = self.network(img_u_w_mix).detach().cpu().float()
                    conf_u_w_mix = (pred_u_w_mix.softmax(dim=1).max(dim=1)[0]).float()
                    mask_u_w_mix = (pred_u_w_mix.argmax(dim=1))
                
                    del img_u_w_mix
                    torch.cuda.empty_cache()
                    
            self.network.train()
            self.optimizer.zero_grad()
            
            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
            img_x, img_u_w,img_u_s1, img_u_s2 = maybe_to_cuda([img_x,img_u_w,img_u_s1,img_u_s2])
            preds, preds_fp = self.network(torch.cat((img_x, img_u_w)), True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]
            
            pred_u_s1, pred_u_s2 = self.network(torch.cat((img_u_s1, img_u_s2))).chunk(2)
            pred_u_w = pred_u_w.detach().cpu().float()
            conf_u_w = (pred_u_w.softmax(dim=1).max(dim=1)[0]).float()
            mask_u_w = pred_u_w.argmax(dim=1)
            
            mask_u_w_cutmixed1, conf_u_w_cutmixed1 = mask_u_w.clone(), conf_u_w.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2 = mask_u_w.clone(), conf_u_w.clone()
            
            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            
            mask_x = mask_x.cuda().long()
            mask_u_w_cutmixed1 = mask_u_w_cutmixed1.cuda().float()
            mask_u_w_cutmixed2 = mask_u_w_cutmixed2.cuda().float()
            mask_u_w = mask_u_w.cuda()
            
            ce_loss = self.criterion_ce(pred_x.float(), mask_x) + 1e-8
            dice_loss = self.criterion_dice(pred_x.softmax(dim=1), mask_x.unsqueeze(1))

            loss_x = (ce_loss+dice_loss)/ 2.0

            loss_u_s1 = self.criterion_dice(pred_u_s1.softmax(dim=1), mask_u_w_cutmixed1.unsqueeze(1),
                                    ignore=(conf_u_w_cutmixed1 < self.conf_thresh))
            
            loss_u_s2 = self.criterion_dice(pred_u_s2.softmax(dim=1), mask_u_w_cutmixed2.unsqueeze(1),
                                    ignore=(conf_u_w_cutmixed2 < self.conf_thresh))
            
            loss_u_w_fp = self.criterion_dice(pred_u_w_fp.softmax(dim=1), mask_u_w.unsqueeze(1),
                                        ignore=(conf_u_w < self.conf_thresh))
            
            l = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0

            if do_backprop:
                l.backward()
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(pred_x, mask_x)

        del mask_x

        return l.detach().cpu().numpy()
    
    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
            data_dict = next(data_generator)
            data = data_dict['data']
            target = data_dict['target']

            data = maybe_to_torch(data)
            target = maybe_to_torch(target)

            if torch.cuda.is_available():
                data = to_cuda(data)
                target = to_cuda(target)

            self.optimizer.zero_grad()

            if self.fp16:
                with autocast():
                    output = self.network(data)
                    del data
                    l = self.loss(output, target)

                if do_backprop:
                    self.amp_grad_scaler.scale(l).backward()
                    self.amp_grad_scaler.step(self.optimizer)
                    self.amp_grad_scaler.update()
            else:
                output = self.network(data)
                del data
                l = self.loss(output, target)

                if do_backprop:
                    l.backward()
                    self.optimizer.step()

            if run_online_evaluation:
                self.run_online_evaluation(output, target)

            del target

            return l.detach().cpu().numpy()