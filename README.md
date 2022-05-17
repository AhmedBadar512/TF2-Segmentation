# TF2-Segmentation
Tensorflow 2 Segmentation with multiple models, multi-gpu support and support for writing tfrecords.

## Creating and Visualizing Dataset
To write tfrecords give path to the image files and label files with the proper unix type path pattern that will pick up recursively the respective images and labels. To write cityscapes you'd do the following.
```
python utils/create_seg_tfrecords.py --dst_path datasets/tf_records/ --dataset_name cityscapes --img_dir datasets/cityscapes/leftImg8bit/train --label_dir datasets/Generic/cityscapes/gtFine/train --img_pattern .png --label_pattern labelIds.png --split train --write
```
Once written you can visualize these using the visualize flag,
```
python utils/create_seg_tfrecords.py --dst_path datasets/tf_records/ --dataset_name cityscapes --img_dir datasets/cityscapes/leftImg8bit/train --label_dir datasets/Generic/cityscapes/gtFine/train --img_pattern .png --label_pattern labelIds.png --split train --visualize
```
## Training Models
Once the TFrecords are written you can read them for training. You can run the following command to train the model. Information on each argument is provided in the description inside the code. Example run,

```python train.py -bs 4 -s output/tf2/ -e 300 -opt Adam -l 50 -c 19 -m deeplabv3 -d cityscapes19 --backbone resnet101v2 --random_crop_max 0.95 --random_crop_min 0.5 -tfrecs datasets/tf_records/```

In current form only 2 models are supported with multiple backbones. New models will be added later.
