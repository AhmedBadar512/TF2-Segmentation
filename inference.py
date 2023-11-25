import os
import tqdm
import tensorflow as tf
import glob
from model_provider import get_model
import argparse
import cv2
from visualization_dicts import gpu_random_labels, generate_random_colors
from dotwiz import DotWizPlus
import yaml


args = argparse.ArgumentParser(description="Train a network with specific settings")
args.add_argument("-cfg", "--config_file", type=str, default=None, help="Select config saved in the model save directory")
args.add_argument("--backbone", type=str, default=None,
                  help="Backbone in case applicable",
                  choices=["resnet50", "resnet101", "resnet152", "xception", "resnet101v2",
                           "resnet50v2",
                           "mobilenetv3large",
                           "mobilenetv3small",
                           "hardnet39",
                           "hardnet68",
                           "hardnet85",
                           ])
args.add_argument("--width", type=int, default=512, help="Size of the shuffle buffer")
args.add_argument("--height", type=int, default=1024, help="Size of the shuffle buffer")
args.add_argument("-c", "--classes", type=int, default=19, help="Number of classes")
args.add_argument("-m", "--model", type=str, default=None, help="Select model")
args.add_argument("-d", "--data_path", type=str, help="Path to folder with images")
args.add_argument("-i", "--infer_path", type=str, default="infer_out/", help="Path to folder with images")
args.add_argument("-l_m", "--load_model", type=str, help="Load model from path")
args = args.parse_args()

if args.config_file is not None:
    cfg_args = DotWizPlus(yaml.safe_load(open(args.config_file)))
    args.backbone = cfg_args.backbone
    args.width = cfg_args.width
    args.height = cfg_args.height
    args.classes = cfg_args.classes
    args.model = cfg_args.model

color_dst, seg_dst = args.infer_path + "/colormaps", args.infer_path + "/segmaps"
os.makedirs(color_dst, exist_ok=True)
os.makedirs(seg_dst, exist_ok=True)
img_pths = glob.glob(f"{args.data_path}/*g")

model = get_model(args.model, classes=args.classes, aux=False, in_size=(args.height, args.width),
                 backbone=args.backbone)
model(tf.random.uniform((1, args.height, args.height, 3)))
model.load_weights(args.load_model + "/saved_model")
cmp = generate_random_colors()
for img_pth in tqdm.tqdm(img_pths):
    img = cv2.cvtColor(cv2.imread(img_pth), cv2.COLOR_BGR2RGB)
    x = tf.image.per_image_standardization(img)
    x = tf.image.resize(x, (args.height, args.width))[tf.newaxis]
    y = tf.argmax(model(x, training=True), -1)
    # y = tf.image.resize(y, (img.shape[1], img.shape[0]))
    color_y = gpu_random_labels(y, cmp)

    cv2.imwrite(f"{color_dst}/{os.path.basename(img_pth)}", cv2.resize(color_y[0].numpy().astype('uint8'), (img.shape[1], img.shape[0])))
    cv2.imwrite(f"{seg_dst}/{os.path.basename(img_pth)}", cv2.resize(y[0].numpy().astype('uint8'), (img.shape[1], img.shape[0]), cv2.INTER_NEAREST))