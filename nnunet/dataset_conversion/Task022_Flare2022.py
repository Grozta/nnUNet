# _*_ coding:utf-8 _*_
import glob
import os
import re
from collections import OrderedDict
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool
import shutil

from batchgenerators.utilities.file_and_folder_operations import save_json

'''
所有的image文件，必须以0000.nii.gz结尾，这里的0000是模态（modality）的识别码
所有的test文件，必须以0000.nii.gz结尾，这里的0000是模态（modality）的识别码
所有的lable文件名必须要和image文件名（去掉0000，即模态标识符之后）相同，
也就是说image和label的标识符要相同，但是image文件名多一个模态识别码
'''

'''
但是，在json文件中，training和test两个字段中只能是不含模态识别码的文件名。
另外其中的路径中最少包含一个/
'''

def list_sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """

    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    l.sort(key=alphanum_key)
    return l

def rename_format(path_originalData):
    train_image_path = os.path.join(path_originalData, 'imagesTr')
    train_label_path = os.path.join(path_originalData, 'labelsTr')

    train_image = os.listdir(train_image_path)
    train_label = os.listdir(train_label_path) 

    for image, label in zip(train_image,train_label):
        image_name_list = image.split('_')
        image_name_list[-2] = '%03d'%int(image_name_list[-2])
        image_name = '_'.join(image_name_list)
        
        image_name_list[-1] = ".nii.gz"
        label_name = image_name_list[0]+'_'+\
                    image_name_list[1]+'_'+\
                    image_name_list[2]+\
                    image_name_list[3]
        
        os.rename(os.path.join(train_image_path,image),os.path.join(train_image_path,image_name))
        os.rename(os.path.join(train_label_path,label),os.path.join(train_label_path,label_name))

def gen_identif(file_base_name):
    """base_name like :Case_00007_0000.nii.gz 
    remove  modality information  0000
       target like: Case_00007.nii.gz
    """
    s = file_base_name.split('_')
    target = s[0]+'_'+ s[1] + '.nii.gz'
    return target


def check_spacing_move(image_path,move_path):
    data_itk = sitk.ReadImage(image_path)
    spacing = np.array(data_itk.GetSpacing())[[2, 1, 0]]
    if spacing[0]>4.5:
        shutil.move(image_path, move_path)
        print(image_path+f'{spacing}'+ 'moveing')

# path_originalData = 'xxxx/PycharmProjects/nnUNet/nnUNet_raw/nnUNet_raw_data/Task072_HCC/'
path_originalData = '/media/icml/HDD/hjc/dataset/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task023_FLARE22/'
#rename_format(path_originalData)
train_image = list_sort_nicely(glob.glob(os.path.join(path_originalData, 'imagesTr', '*')))
train_label = list_sort_nicely(glob.glob(os.path.join(path_originalData, 'labelsTr', '*')))
test_image = list_sort_nicely(glob.glob(os.path.join(path_originalData, 'imagesTs', '*')))
test_label = list_sort_nicely(glob.glob(os.path.join(path_originalData, 'labelsTs', '*')))
unlabeled_image = list_sort_nicely(glob.glob(os.path.join(path_originalData, 'unlabeled', '*.gz')))

p = Pool(10)
p.starmap_async(check_spacing_move, [(image_path, os.path.join(path_originalData, 'unlabeled00')) for image_path in unlabeled_image])
p.close()
p.join()

unlabeled_image = list_sort_nicely(glob.glob(os.path.join(path_originalData, 'unlabeled', '*.gz')))

train_image = ["{}".format(os.path.basename(item)) for item in train_image]
train_label = ["{}".format(os.path.basename(item)) for item in train_label]
test_image = ["{}".format(os.path.basename(item)) for item in test_image]
test_label = ["{}".format(os.path.basename(item)) for item in test_label]
unlabeled_image = ["{}".format(gen_identif(os.path.basename(item))) for item in unlabeled_image]

if len(unlabeled_image)>200:
    unlabeled_image = unlabeled_image[:200]

# 输出一下目录的情况，看是否成功
print(train_image)
print(train_label)
print(test_image)
print(test_label)

# 自行修改
json_dict = OrderedDict()
json_dict['name'] = "FLARE22"
json_dict['description'] = "Task023 for semi-supervised include FLARE22 and unlabeled image"
json_dict['tensorImageSize'] = "3D"
json_dict['reference'] = "Task023"
json_dict['licence'] = "semi-supervised"
json_dict['release'] = "0.1"
json_dict['modality'] = {
    "0": "CT"
    # "0": "HBP",
    # "1" : "T1"
    # 将模态信息写在这里

}
json_dict['labels'] = {
    "0": "background",
    "1": "Liver",
    "2": "Right kidney",
    "3": "Spleen",
    "4": "Pancreas",
    "5": "Aorta",
    "6": "Inferior vena cava",
    "7": "Right adrenal gland",
    "8": "Left adrenal gland",
    "9": "Gallbladder",
    "10": "Esophagus",
    "11": "Stomach",
    "12": "Duodenum",
    "13": "Left Kidney",
    #"14": "Tumor",
}
json_dict['numUnlabeled'] = len(unlabeled_image)
json_dict['unlabeled'] = [{'image': f"./unlabeled/{i}" , "label": None} for i in unlabeled_image]
json_dict['numTraining'] = len(train_image)
json_dict['training'] = [{'image': f"./imagesTr/{i}" , "label": f"./labelsTr/{i}"} for i in train_label]
json_dict['numTest'] = len(test_image)
json_dict['test'] = [{'image': f"./imagesTs/{i}" , "label": f"./labelsTs/{i}"} for i in test_label]
save_json(json_dict, os.path.join(path_originalData, "dataset.json"))