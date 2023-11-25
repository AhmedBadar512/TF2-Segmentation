import tensorflow as tf
from model_provider import get_model
import argparse
from dotwiz import DotWizPlus
import yaml
import tf2onnx


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
args.add_argument("-onnx", "--onnx_path", type=str, help="Convert and save onnx here")
args.add_argument("-l_m", "--load_model", type=str, help="Load model from path")
args = args.parse_args()

if args.config_file is not None:
    cfg_args = DotWizPlus(yaml.safe_load(open(args.config_file)))
    args.backbone = cfg_args.backbone
    args.width = cfg_args.width
    args.height = cfg_args.height
    args.classes = cfg_args.classes
    args.model = cfg_args.model

model = get_model(args.model, classes=args.classes, aux=False, in_size=(args.height, args.width),
                 backbone=args.backbone)
model(tf.random.uniform((1, args.height, args.height, 3)))
model.load_weights(args.load_model + "/saved_model")
spec = tf.TensorSpec((1, args.height, args.width, 3), tf.float32, name="input")
tf2onnx.convert.from_keras(model, [spec], output_path=args.onnx_path)