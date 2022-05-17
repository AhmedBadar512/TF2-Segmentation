import tensorflow as tf


def augment_seg(image, label,
                v_flip=False,
                h_flip=False,
                crop_scale=(0.05, 0.95),
                rand_hue=False,
                rand_sat=False,
                rand_brightness=False,
                rand_contrast=False,
                rand_quality=False):
    if h_flip:
        image = tf.image.random_flip_left_right(image, seed=0)
        label = tf.image.random_flip_left_right(label, seed=0)
    if v_flip:
        image = tf.image.random_flip_up_down(image, seed=0)
        label = tf.image.random_flip_up_down(label, seed=0)
    if crop_scale is not None:
        img_shp = tf.cast(tf.shape(image), tf.float32)
        h_scale = tf.random.uniform([], crop_scale[0], crop_scale[1]) * img_shp[0]
        w_scale = tf.random.uniform([], crop_scale[0], crop_scale[1]) * img_shp[1]
        image_crop = [h_scale, w_scale, image.shape[-1]]
        label_crop = [h_scale, w_scale, label.shape[-1]]
        image = tf.image.random_crop(image, image_crop, seed=0)
        label = tf.image.random_crop(label, label_crop, seed=0)
    if rand_brightness:
        image = tf.image.random_brightness(image, 0.2)
    if rand_hue:
        image = tf.image.random_hue(image, 0.2)
    if rand_sat:
        image = tf.image.random_saturation(image, 1, 1.5)
    if rand_contrast:
        image = tf.image.random_contrast(image, 0.8, 1.2)
    if rand_quality:
        image = tf.image.random_jpeg_quality(image, 50, 100)
    return image, label
