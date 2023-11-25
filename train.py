import argparse
import datetime
import json
import os
import string

import tensorflow as tf
import keras
import tqdm

import losses
import utils.augment_images as aug
from visualizer import get_images_custom
from model_provider import get_model
from utils.create_seg_tfrecords import TFRecordsSeg
from visualization_dicts import gpu_cs_labels, generate_random_colors, gpu_random_labels

EPSILON = 1e-6


def list_of_ints(arg):
    return tuple(map(int, arg.split(',')))


args = argparse.ArgumentParser(description="Train a network with specific settings")
args.add_argument("--backbone", type=str, default="",
                  help="Backbone in case applicable",
                  choices=["resnet50", "resnet101", "resnet152", "xception", "resnet101v2",
                           "resnet50v2",
                           "mobilenetv3large",
                           "mobilenetv3small",
                           "hardnet39",
                           "hardnet68",
                           "hardnet85",
                           ])
args.add_argument("-d", "--dataset", type=str, default="cityscapes19",
                  help="Name a dataset from the tf_dataset collection",
                  choices=["cityscapes", "cityscapes19", "ade20k", "fangzhou", "bdd_drivable"])
args.add_argument("-c", "--classes", type=int, default=19, help="Number of classes")
args.add_argument("-opt", "--optimizer", type=str, default="Adam", help="Select optimizer",
                  choices=["SGD", "RMSProp", "Adam"])
args.add_argument("-lrs", "--lr_scheduler", type=str, default="exp_decay", help="Select learning rate scheduler",
                  choices=["poly", "exp_decay", "cos_decay_r", "constant"])
args.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs to train")
args.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
args.add_argument("--momentum", type=float, default=0.9, help="Momentum")
args.add_argument("-l", "--logging_freq", type=int, default=10, help="Add to tfrecords after this many steps")
args.add_argument("--loss", type=str, default="cross_entropy",
                  choices=["cross_entropy", "focal_loss", "binary_crossentropy", "RMI"],
                  help="Loss function")
args.add_argument("-bs", "--batch_size", type=int, default=4, help="Size of mini-batch")
args.add_argument("-si", "--save_interval", type=int, default=5, help="Save interval for model")
args.add_argument("-wis", "--write_image_summary_steps", type=int, default=50, help="Add images to tfrecords "

                                                                                    "after these many logging steps")
args.add_argument("-m", "--model", type=str, default="bisenetv2", help="Select model")
args.add_argument("-l_m", "--load_model", type=str,
                  default=None,
                  help="Load model from path")
args.add_argument("-s", "--save_dir", type=str, default="./runs", help="Save directory for models and tensorboard")
args.add_argument("-tfrecs", "--tf_record_path", type=str, default="/data/input-ai/datasets/tf2_segmentation_tfrecords",
                  help="Save directory that contains train and validation tfrecords")
args.add_argument("-sb", "--shuffle_buffer", type=int, default=128, help="Size of the shuffle buffer")
args.add_argument("--width", type=int, default=512, help="Size of the shuffle buffer")
args.add_argument("--height", type=int, default=512, help="Size of the shuffle buffer")
args.add_argument("--aux", action="store_true", default=False, help="Auxiliary losses included if true")
args.add_argument("--aux_weight", type=float, default=0.2, help="Auxiliary losses included if true")
args.add_argument("--random_seed", type=int, default=512, help="Set random seed to this if true")
args.add_argument("--bg_class", type=int, default=0, help="Select bg class for visualization shown as black")
args.add_argument("--fp16", action="store_true", default=False, help="Give to enable mixed precision training.")
# ============ Augmentation Arguments ===================== #
args.add_argument("--flip_up_down", action="store_true", default=False, help="Randomly flip images up and down")
args.add_argument("--flip_left_right", action="store_true", default=False, help="Randomly flip images right left")
args.add_argument("--random_crop", type=list_of_ints, default=None,
                  help="Pass a tuple as h,w for size of random crop e.g. 512,512 to crop shape (512, 512)")
args.add_argument("--random_hue", action="store_true", default=False, help="Randomly change hue")
args.add_argument("--random_saturation", action="store_true", default=False, help="Randomly change saturation")
args.add_argument("--random_brightness", action="store_true", default=False, help="Randomly change brightness")
args.add_argument("--random_contrast", action="store_true", default=False, help="Randomly change contrast")
args.add_argument("--random_quality", action="store_true", default=False, help="Randomly change jpeg quality")
args.add_argument("--all_augs", action="store_true", default=False, help="Add all augmentations except flip_up_down")
args = args.parse_args()

if args.fp16:
    keras.mixed_precision.set_global_policy('mixed_float16')

physical_devices = tf.config.experimental.list_physical_devices("GPU")
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
mirrored_strategy = tf.distribute.MirroredStrategy()

tf.random.set_seed(args.random_seed)
random_crop_size = args.random_crop
backbone = args.backbone
dataset_name = args.dataset
aux = args.aux
aux_weight = args.aux_weight
epochs = args.epochs
batch_size = args.batch_size
classes = args.classes
optimizer_name = args.optimizer
lr = args.lr
momentum = args.momentum
model_name = args.model
if model_name in ['bisenetv2']:
    aux = True
log_freq = args.logging_freq
write_image_summary_steps = args.write_image_summary_steps
EPOCHS = args.epochs
time = str(datetime.datetime.now())
time = time.translate(str.maketrans('', '', string.punctuation)).replace(" ", "-")[:-8]
logdir = os.path.join(args.save_dir,
                      "{}_epochs-{}_{}_bs-{}_{}_lr_{}-{}_{}_{}_{}".format(dataset_name, epochs, args.loss,
                                                                          batch_size,
                                                                          optimizer_name, lr,
                                                                          args.lr_scheduler,
                                                                          backbone,
                                                                          model_name,
                                                                          time))

# =========== Load Dataset ============ #
cmap = generate_random_colors(bg_class=args.bg_class)

dataset_train = TFRecordsSeg(
    tfrecord_path=
    "{}/{}_train.tfrecords".format(args.tf_record_path, dataset_name)).read_tfrecords()
dataset_validation = TFRecordsSeg(
    tfrecord_path=
    "{}/{}_val.tfrecords".format(args.tf_record_path, dataset_name)).read_tfrecords()

if args.all_augs:
    args.flip_left_right = True
    random_crop_size = (512, 512)
    args.random_hue = True
    args.random_saturation = True
    args.random_brightness = True
    args.random_contrast = True

augmentor = lambda image, label: aug.augment_seg(image, label,
                                                 args.flip_up_down,
                                                 args.flip_left_right,
                                                 random_crop_size,
                                                 args.random_hue,
                                                 args.random_saturation,
                                                 args.random_brightness,
                                                 args.random_contrast,
                                                 args.random_quality)
with open(f"{args.tf_record_path}/data_samples.json") as f:
    data = json.load(f)
total_samples = data[dataset_name]
# =========== Process dataset ============ #
assert dataset_train is not None, "Training dataset can not be None"
assert dataset_validation is not None, "Either test or validation dataset should not be None"

eval_dataset = dataset_validation
get_images_processed = lambda image, label: get_images_custom(image, label, (args.height, args.width))

processed_train = dataset_train.map(augmentor)
processed_train = processed_train.map(get_images_processed)
processed_val = dataset_validation.map(get_images_processed)
processed_train = processed_train.batch(batch_size, drop_remainder=True).repeat(
    EPOCHS).shuffle(args.shuffle_buffer).prefetch(
    tf.data.experimental.AUTOTUNE)
processed_val = processed_val.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE) \
    if (dataset_validation is not None) else None
processed_train = mirrored_strategy.experimental_distribute_dataset(processed_train)
processed_val = mirrored_strategy.experimental_distribute_dataset(processed_val)
# =========== Optimizer and Training Setup ============ #
with mirrored_strategy.scope():
    if args.lr_scheduler == "poly":
        lr_scheduler = keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=lr,
                                                                     decay_steps=epochs * total_samples // batch_size,
                                                                     end_learning_rate=1e-12,
                                                                     power=0.9)
    elif args.lr_scheduler == "exp_decay":
        lr_scheduler = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr,
                                                                      decay_steps=epochs * total_samples // batch_size,
                                                                      decay_rate=0.9)
    elif args.lr_scheduler == "cos_decay_r":
        lr_scheduler = keras.optimizers.schedules.CosineDecayRestarts(lr, (1.1 * total_samples) // batch_size, 2.0, 0.98, 0.01)
    else:
        lr_scheduler = lr

    if optimizer_name == "Adam":
        optimizer = keras.optimizers.Adam(learning_rate=lr_scheduler)
    elif optimizer_name == "RMSProp":
        optimizer = keras.optimizers.RMSprop(learning_rate=lr_scheduler, momentum=momentum)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=lr_scheduler, momentum=momentum)
    model = get_model(model_name, classes=classes, in_size=(args.height, args.width), aux=aux,
                      backbone=args.backbone)
    model(tf.random.uniform((1, args.height, args.width, 3), dtype=tf.float32), True, aux=True)
    model.summary()
    if args.load_model:
        if os.path.exists(os.path.join(args.load_model)):
            # pretrained_model = K.models.load_model(args.load_model)
            model.load_weights(os.path.join(args.load_model, "saved_model"))
            print("Model loaded from {} successfully".format(os.path.basename(args.load_model)))
        else:
            print("No file found at {}".format(os.path.join(args.load_model, "saved_model")))

calc_loss = losses.get_loss(name=args.loss)


def train_step(mini_batch, aux=False, pick=None):
    with tf.GradientTape() as tape:
        train_logits = model(tf.image.per_image_standardization(mini_batch[0]), training=True, aux=aux)
        train_labs = tf.one_hot(mini_batch[1][..., 0], classes) + EPSILON
        loss_mask = tf.reduce_sum(train_labs, axis=-1)
        if aux:
            losses = [loss_mask * calc_loss(train_labs, tf.image.resize(train_logit, size=train_labs.shape[
                                                                                             1:3])) if n == 0 else args.aux_weight *
                loss_mask * calc_loss(
                    train_labs, tf.image.resize(train_logit, size=train_labs.shape[1:3])) for n, train_logit in
                      enumerate(train_logits)]
            loss = sum(losses)
            train_logits = train_logits[0]
        else:
            loss = loss_mask * calc_loss(train_labs, train_logits)
        loss = tf.reduce_mean(loss)
    if pick is not None:
        trainable_vars = [var for var in model.trainable_variables if pick in var.name]
    else:
        trainable_vars = model.trainable_variables
    grads = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(grads, trainable_vars))
    return loss, train_labs, tf.image.resize(train_logits, tf.shape(train_labs)[1:3],
                                             method=tf.image.ResizeMethod.BILINEAR)


def val_step(mini_batch, aux=False):
    val_logits = model(tf.image.per_image_standardization(mini_batch[0]),
                       training=True, aux=aux)
    val_labs = tf.one_hot(mini_batch[1][..., 0], classes) + EPSILON
    loss_mask = tf.reduce_sum(val_labs, axis=-1)
    if aux:
        losses = [loss_mask * calc_loss(val_labs, tf.image.resize(train_logit, size=val_labs.shape[1:3])) if n == 0
                  else args.aux_weight * loss_mask * calc_loss(val_labs, tf.image.resize(train_logit, size=val_labs.shape[1:3]))
                  for n, train_logit in enumerate(val_logits)]
        val_loss = sum(losses)
        val_logits = val_logits[0]
    else:
        val_loss = loss_mask * calc_loss(val_labs, val_logits)
    val_loss = tf.reduce_mean(val_loss)
    return val_loss, val_labs, tf.image.resize(val_logits, tf.shape(val_labs)[1:3],
                                               method=tf.image.ResizeMethod.BILINEAR)


@tf.function
def distributed_train_step(dist_inputs):
    per_replica_losses, train_labs, train_logits = mirrored_strategy.run(train_step, args=(dist_inputs, aux))
    loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                    axis=None)
    if len(physical_devices) > 1:
        return loss, \
               train_labs.values[0], \
               train_logits.values[0]
    else:
        return loss, \
               train_labs, \
               train_logits


@tf.function
def distributed_val_step(dist_inputs):
    per_replica_losses, val_labs, val_logits = mirrored_strategy.run(val_step, args=(dist_inputs, aux))
    loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                    axis=None)
    if len(physical_devices) > 1:
        return loss, \
               val_labs.values[0], \
               val_logits.values[0]
    else:
        return loss, \
               val_labs, \
               val_logits


# =========== Training ============ #


train_writer = tf.summary.create_file_writer(os.path.join(logdir, "train"))
val_writer = tf.summary.create_file_writer(os.path.join(logdir, "val"))

mIoU = keras.metrics.MeanIoU(classes)

if args.load_model is not None:
    START_EPOCH = int(args.load_model.split("/")[-1])
else:
    START_EPOCH = 0


def write_summary_images(batch, logits):
    if len(physical_devices) > 1:
        tf.summary.image("images", batch[0].values[0] / 255, step=c_step)
        processed_labs = batch[1].values[0]
    else:
        tf.summary.image("images", batch[0] / 255, step=c_step)
        processed_labs = batch[1]
    if dataset_name == "cityscapes19":
        colorize = lambda img: tf.cast(tf.squeeze(gpu_cs_labels(img)), dtype=tf.uint8)
        tf.summary.image("pred", colorize(tf.argmax(logits, axis=-1)), step=c_step)
        tf.summary.image("gt", colorize(processed_labs[..., tf.newaxis]), step=c_step)
    else:
        colorize = lambda img: tf.cast(tf.squeeze(gpu_random_labels(img, cmap)), dtype=tf.uint8)
        tf.summary.image("pred", colorize(tf.argmax(logits, axis=-1)), step=c_step)
        tf.summary.image("gt", colorize(processed_labs[..., tf.newaxis]), step=c_step)


mini_batch, train_logits = None, None
val_mini_batch, val_logits = None, None
image_write_step = 0
c_step = 0


def write_to_tensorboard(curr_step, image_write_step, writer, logits, batch):
    if curr_step % log_freq == 0:
        image_write_step += 1
        with writer.as_default():
            tf.summary.scalar("loss", loss,
                              step=curr_step)
            if curr_step % write_image_summary_steps == 0:
                write_summary_images(batch, logits)
    with writer.as_default():
        tmp = lr_scheduler(step=curr_step)
        tf.summary.scalar("Learning Rate", tmp, curr_step)


def evaluate():
    global val_writer
    mIoU.reset_states()
    conf_matrix_list = []
    total_val_loss = []
    val_steps = 1
    for val_steps, val_mini_batch in tqdm.tqdm(enumerate(processed_val)):
        loss, val_labs, val_logits = distributed_val_step(val_mini_batch)
        gt = tf.reshape(tf.argmax(val_labs, axis=-1), -1)
        pred = tf.reshape(tf.argmax(val_logits, axis=-1), -1)
        mIoU.update_state(gt, pred)
        conf_matrix = tf.math.confusion_matrix(gt, pred,
                                               num_classes=classes)
        conf_matrix_list.append(conf_matrix)
        total_val_loss.append(tf.reduce_mean(calc_loss(val_labs, val_logits)))
    val_loss = tf.reduce_mean(total_val_loss)
    conf_matrix = tf.reduce_sum(conf_matrix_list, axis=0) / (batch_size * val_steps)
    with val_writer.as_default():
        tf.summary.scalar("loss", val_loss,
                          step=c_step)
        tf.summary.scalar("mIoU", mIoU.result().numpy(),
                          step=c_step)
        if val_mini_batch is not None:
            conf_matrix = tf.cast(conf_matrix, dtype=tf.float64) / (
                    tf.cast(tf.reduce_sum(conf_matrix, axis=1), dtype=tf.float64) + 1e-6)
            tf.summary.image("conf_matrix", conf_matrix[tf.newaxis, ..., tf.newaxis], step=c_step)
            write_summary_images(val_mini_batch, val_logits)
    print("Val Epoch {}: {}, mIoU: {}".format(epoch, val_loss, mIoU.result().numpy()))


epoch = 0
while epoch < EPOCHS:
    print("\n ----------- Epoch {} --------------\n".format(epoch))
    step = 0
    if epoch % args.save_interval == 0:
        model.save_weights(os.path.join(logdir, model_name, str(epoch), "saved_model"))
        print("Model at Epoch {}, saved at {}".format(epoch, os.path.join(logdir, model_name, str(epoch))))
    for mini_batch in tqdm.tqdm(processed_train, total=total_samples // args.batch_size):
        c_step = (epoch * total_samples // args.batch_size) + step
        loss, train_labs, train_logits = distributed_train_step(mini_batch)
        step += 1

        # ====================================
        write_to_tensorboard(c_step, image_write_step, train_writer, train_logits, mini_batch)
        if step == total_samples // args.batch_size:
            epoch += 1
            break

    evaluate()
