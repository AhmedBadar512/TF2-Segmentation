import tensorflow as tf
import keras
from keras.layers import Conv2D
from models.backbones import get_backbone, get_bb_dict, get_custom_bb_dict
from models.layers import ConvBlock


class AtrousSpatialPyramidPooling(keras.layers.Layer):
    """Atrous Spatial Pyramid Pooling layer for DeepLabV3+ architecture."""

    # !pylint:disable=too-many-instance-attributes
    def __init__(self):
        super(AtrousSpatialPyramidPooling, self).__init__()

        # layer architecture components
        self.avg_pool = None
        self.conv1, self.conv2 = None, None
        self.pool = None
        self.out1, self.out6, self.out12, self.out18 = None, None, None, None
        self.concat = keras.layers.Concatenate(axis=-1)

    @staticmethod
    def _get_conv_block(kernel_size, dilation_rate, use_bias=False):
        return ConvBlock(256,
                         kernel_size=kernel_size,
                         dilation_rate=dilation_rate,
                         padding='same',
                         use_bias=use_bias,
                         kernel_initializer=keras.initializers.he_normal(),
                         activation='relu')

    def build(self, input_shape):
        dummy_tensor = tf.random.normal(input_shape)  # used for calculating
        # output shape of convolutional layers

        self.avg_pool = keras.layers.AveragePooling2D(
            pool_size=(input_shape[-3], input_shape[-2]))

        self.conv1 = AtrousSpatialPyramidPooling._get_conv_block(
            kernel_size=1, dilation_rate=1, use_bias=True)

        self.conv2 = AtrousSpatialPyramidPooling._get_conv_block(
            kernel_size=1, dilation_rate=1)

        dummy_tensor = self.conv1(self.avg_pool(dummy_tensor))

        self.pool = keras.layers.UpSampling2D(
            size=(
                input_shape[-3] // dummy_tensor.shape[1],
                input_shape[-2] // dummy_tensor.shape[2]
            ),
            interpolation='bilinear'
        )

        self.out1, self.out6, self.out12, self.out18 = map(
            lambda tup: AtrousSpatialPyramidPooling._get_conv_block(
                kernel_size=tup[0], dilation_rate=tup[1]
            ),
            [(1, 1), (3, 6), (3, 12), (3, 18)]
        )

    def call(self, inputs, training=None, **kwargs):
        tensor = self.avg_pool(inputs)
        tensor = self.conv1(tensor)
        tensor = self.concat([
            self.pool(tensor),
            self.out1(inputs),
            self.out6(inputs),
            self.out12(
                inputs
            ),
            self.out18(
                inputs
            )
        ])
        tensor = self.conv2(tensor)
        return tensor


class Deeplabv3plus(keras.Model):
    def __init__(self, backbone="resnet50", classes=19, activation='relu', **kwargs):
        super().__init__()
        self.backbone_name = backbone
        self.aspp = AtrousSpatialPyramidPooling()
        self.convblock1 = ConvBlock(48, 1, padding="same", use_bias=False, activation=activation)
        self.convblock2 = ConvBlock(256, 3, padding="same", use_bias=False, activation=activation)
        self.convblock3 = ConvBlock(256, 3, padding="same", use_bias=False, activation=activation)
        self.conv_f = Conv2D(classes, (1, 1), name='output_layer')

    def build(self, input_shape):
        self.backbone, base_backbone = get_backbone(self.backbone_name, input_shape=input_shape[1:])
        aspp_shape, backbone_b_shape = ((input_shape[1] // 4, input_shape[2] // 4),
                                        (input_shape[1] // 16, input_shape[2] // 16))
        if base_backbone:
            ind_dict = get_bb_dict(self.backbone, divisible_factor=32)
            aspp_index, backbone_b_index = ind_dict[aspp_shape][-1], ind_dict[backbone_b_shape][-1]
            self.backbone = keras.Model(inputs=self.backbone.input, outputs=[self.backbone.layers[aspp_index].output, self.backbone.layers[backbone_b_index].output])
        else:
            outs = self.backbone(keras.Input(input_shape[1:]))
            out_dict = get_custom_bb_dict(outs, 32)
            self.backbone = keras.Model(self.backbone.input,
                                        outputs=[outs[out_dict[aspp_shape]], outs[out_dict[backbone_b_shape]]])


    def call(self, inputs, training=None, mask=None, aux=False):
        x_aspp, x_b = self.backbone(inputs)
        x_aspp = self.aspp(x_aspp, training)
        x_a = tf.image.resize(x_aspp, tf.shape(x_b)[1:3])
        x_b = self.convblock1(x_b, training)
        x_ab = keras.layers.concatenate([x_a, x_b])
        x_ab = self.convblock2(x_ab, training)
        x_ab = self.convblock3(x_ab, training)
        x_ab = tf.image.resize(x_ab, tf.shape(inputs)[1:3])
        return self.conv_f(x_ab)


if __name__ == "__main__":
    i = tf.random.uniform((1, 512, 512, 3))
    deeplab_model = Deeplabv3plus()
    a = deeplab_model(i)
    print(a.shape)
