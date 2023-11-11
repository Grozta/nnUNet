#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.augmentations.spatial_transformations import augment_spatial, augment_spatial_2, \
    augment_channel_translation, \
    augment_mirroring, augment_transpose_axes, augment_zoom, augment_resize, augment_rot90
from batchgenerators.augmentations.color_augmentations import augment_gamma
from copy import deepcopy
import torch
import random


class RemoveKeyTransform(AbstractTransform):
    def __init__(self, key_to_remove):
        self.key_to_remove = key_to_remove

    def __call__(self, **data_dict):
        _ = data_dict.pop(self.key_to_remove, None)
        return data_dict


class MaskTransform(AbstractTransform):
    def __init__(self, dct_for_where_it_was_used, mask_idx_in_seg=1, set_outside_to=0, data_key="data", seg_key="seg"):
        """
        data[mask < 0] = 0
        Sets everything outside the mask to 0. CAREFUL! outside is defined as < 0, not =0 (in the Mask)!!!

        :param dct_for_where_it_was_used:
        :param mask_idx_in_seg:
        :param set_outside_to:
        :param data_key:
        :param seg_key:
        """
        self.dct_for_where_it_was_used = dct_for_where_it_was_used
        self.seg_key = seg_key
        self.data_key = data_key
        self.set_outside_to = set_outside_to
        self.mask_idx_in_seg = mask_idx_in_seg

    def __call__(self, **data_dict):
        seg = data_dict.get(self.seg_key)
        if seg is None or seg.shape[1] < self.mask_idx_in_seg:
            raise Warning("mask not found, seg may be missing or seg[:, mask_idx_in_seg] may not exist")
        data = data_dict.get(self.data_key)
        for b in range(data.shape[0]):
            mask = seg[b, self.mask_idx_in_seg]
            for c in range(data.shape[1]):
                if self.dct_for_where_it_was_used[c]:
                    data[b, c][mask < 0] = self.set_outside_to
        data_dict[self.data_key] = data
        return data_dict


def convert_3d_to_2d_generator(data_dict):
    shp = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_data'] = shp
    shp = data_dict['seg'].shape
    data_dict['seg'] = data_dict['seg'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_seg'] = shp
    return data_dict


def convert_2d_to_3d_generator(data_dict):
    shp = data_dict['orig_shape_data']
    current_shape = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1], shp[2], current_shape[-2], current_shape[-1]))
    shp = data_dict['orig_shape_seg']
    current_shape_seg = data_dict['seg'].shape
    data_dict['seg'] = data_dict['seg'].reshape((shp[0], shp[1], shp[2], current_shape_seg[-2], current_shape_seg[-1]))
    return data_dict


class Convert3DTo2DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_3d_to_2d_generator(data_dict)


class Convert2DTo3DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_2d_to_3d_generator(data_dict)


class ConvertSegmentationToRegionsTransform(AbstractTransform):
    def __init__(self, regions: dict, seg_key: str = "seg", output_key: str = "seg", seg_channel: int = 0):
        """
        regions are tuple of tuples where each inner tuple holds the class indices that are merged into one region, example:
        regions= ((1, 2), (2, )) will result in 2 regions: one covering the region of labels 1&2 and the other just 2
        :param regions:
        :param seg_key:
        :param output_key:
        """
        self.seg_channel = seg_channel
        self.output_key = output_key
        self.seg_key = seg_key
        self.regions = regions

    def __call__(self, **data_dict):
        seg = data_dict.get(self.seg_key)
        num_regions = len(self.regions)
        if seg is not None:
            seg_shp = seg.shape
            output_shape = list(seg_shp)
            output_shape[1] = num_regions
            region_output = np.zeros(output_shape, dtype=seg.dtype)
            for b in range(seg_shp[0]):
                for r, k in enumerate(self.regions.keys()):
                    for l in self.regions[k]:
                        region_output[b, r][seg[b, self.seg_channel] == l] = 1
            data_dict[self.output_key] = region_output
        return data_dict


class SpatialTransform_for_unimatch(SpatialTransform):
    def __init__(self, patch_size, patch_center_dist_from_border=30,
                 do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                 do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                 do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                 border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, data_key="data",
                 label_key="seg", p_el_per_sample=1, p_scale_per_sample=1, p_rot_per_sample=1,
                 independent_scale_for_each_axis=False, p_rot_per_axis:float=1, p_independent_scale_per_axis: int=1):
        super().__init__(patch_size, patch_center_dist_from_border,
                 do_elastic_deform, alpha, sigma,
                 do_rotation, angle_x, angle_y, angle_z,
                 do_scale, scale, border_mode_data, border_cval_data, order_data,
                 border_mode_seg, border_cval_seg, order_seg, random_crop, data_key,
                 label_key, p_el_per_sample, p_scale_per_sample, p_rot_per_sample,
                 independent_scale_for_each_axis, p_rot_per_axis, p_independent_scale_per_axis)
        self.image_indx_list = ['image_u_w','image_u_w_mix']
    def __call__(self, **data_dict):
        """
        返回的就是data_dict
        return {'image_u_w': unlabeled_1, 'image_u_w_mix': unlabeled_2,
                'image_u_s1': None,'image_u_s2': None,
                "mixcut_box_1": None,"mixcut_box_2":None,
                "image_identifier":image_identifier}
        """
        patch_size = self.patch_size
            
        for image_indx in self.image_indx_list:
            ret = augment_spatial(data_dict[image_indx], None, patch_size=patch_size,
                                    patch_center_dist_from_border=self.patch_center_dist_from_border,
                                    do_elastic_deform=self.do_elastic_deform, alpha=self.alpha, sigma=self.sigma,
                                    do_rotation=self.do_rotation, angle_x=self.angle_x, angle_y=self.angle_y,
                                    angle_z=self.angle_z, do_scale=self.do_scale, scale=self.scale,
                                    border_mode_data=self.border_mode_data,
                                    border_cval_data=self.border_cval_data, order_data=self.order_data,
                                    border_mode_seg=self.border_mode_seg, border_cval_seg=self.border_cval_seg,
                                    order_seg=self.order_seg, random_crop=self.random_crop,
                                    p_el_per_sample=self.p_el_per_sample, p_scale_per_sample=self.p_scale_per_sample,
                                    p_rot_per_sample=self.p_rot_per_sample,
                                    independent_scale_for_each_axis=self.independent_scale_for_each_axis,
                                    p_rot_per_axis=self.p_rot_per_axis, 
                                    p_independent_scale_per_axis=self.p_independent_scale_per_axis)
            data_dict[image_indx] = ret[0]
        #print('SpatialTransform_for_unimatch is over')
        return data_dict
        
class CloneAndCutmixTransform_for_unimatch(AbstractTransform):
    
    def __init__(self):
        """
        对空间增强过的图像clone出 s1 和 s2
        """
        self.image_indx_list = ['image_u_w','image_u_w_mix']
        
    
    def obtain_cutmix_box_3d(self, img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
        mask = torch.zeros(img_size)
        if random.random() > p:
            return mask

        size = np.random.uniform(size_min, size_max) * img_size[0] * img_size[1] * img_size[2]
        while True:
            ratio = np.random.uniform(ratio_1, ratio_2)
            cutmix_d = int(np.cbrt(size * ratio))
            cutmix_w = int(np.cbrt (size / ratio))
            cutmix_h = int(np.cbrt(size / ratio))
            d = np.random.randint(0, img_size[0])
            w = np.random.randint(0, img_size[1])
            h = np.random.randint(0, img_size[2])

            if w + cutmix_w <= img_size[0] and h + cutmix_h <= img_size[1] and d + cutmix_d <= img_size[2]:
                break

        mask[ d:d+cutmix_d,w:w + cutmix_w, h:h + cutmix_h] = 1
        return mask

    def __call__(self, **data_dict):
        """
        返回的就是data_dict
        return {'image_u_w': unlabeled_1, 'image_u_w_mix': unlabeled_2,
                'image_u_s1': None,'image_u_s2': None,
                "mixcut_box_1": None,"mixcut_box_2":None,
                "image_identifier":image_identifier}
        """
        
        res = []
        # 强增强
        for image_indx in self.image_indx_list:
            img_s1, img_s2 = deepcopy(data_dict[image_indx]), deepcopy(data_dict[image_indx])
            img_s1, img_s2 = augment_gamma(img_s1),augment_gamma(img_s2)
            
            img_s1= augment_mirroring(np.squeeze(img_s1, axis=0))
            img_s2 =augment_mirroring(np.squeeze(img_s2, axis=0))
            res.append([img_s1[0],img_s1[0]])
        # cutmix处理
        size = list((data_dict["image_u_w"].shape)[2:])
        cutmix_box1, cutmix_box2 = self.obtain_cutmix_box_3d(size), self.obtain_cutmix_box_3d(size)
        img_u_s1,img_u_s1_mix,img_u_s2,img_u_s2_mix = res[0][0],res[1][0],res[0][1],res[1][1]
        img_u_s1[cutmix_box1.unsqueeze(0).expand(img_u_s1.shape) == 1] = \
            img_u_s1_mix[cutmix_box1.unsqueeze(0).expand(img_u_s1.shape) == 1]
        img_u_s2[cutmix_box2.unsqueeze(0).expand(img_u_s2.shape) == 1] = \
            img_u_s2_mix[cutmix_box2.unsqueeze(0).expand(img_u_s2.shape) == 1]
        
        data_dict["image_u_s1"], data_dict["image_u_s2"] = torch.from_numpy(img_u_s1).unsqueeze(0), torch.from_numpy(img_u_s2).unsqueeze(0)
        data_dict["mixcut_box_1"], data_dict["mixcut_box_2"] = cutmix_box1.unsqueeze(0), cutmix_box2.unsqueeze(0)
        return data_dict
            