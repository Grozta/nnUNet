# unimatch 在nnunet上运行

## 数据部分

有标签数据是Flare22上的50例有标签的数据和2000例无标签的数据。但是nnunet不能处理无标签的数据，所以第一步就是处理无标签的数据。

### 数据预处理

使用训练好的模型对2000个无标签数据进行识别，只切割保留腹部图像。

1. 使用早些在Task022_FLARE22任务中的全监督模型，对无标签进行预测

   - ```bash
     nnUNet_predict -i INPUTS_FOLDER -o OUTPUTS_FOLDER  -t 22  -tr nnUNetTrainerV2_FLARE_Big  -m 3d_fullres  -p nnUNetPlansFLARE22Big  --all_in_gpu True
     ```
   - 
2. 对mask进行分析，获取他的bbox，然后对bbox进行膨胀
3. 使用bbox对原图进行切割，保留整个过程的记录pkl和新的图像数据nzp

### 计划生成

### 修改dataload

### 确定最佳patch-size
