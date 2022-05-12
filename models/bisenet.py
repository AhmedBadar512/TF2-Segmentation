import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
import tensorflow.keras as K
from models.backbones import get_backbone
from models.layers import ConvBlock, StemBlock, DWConvBlock


class ContextEmbeddingBlock(K.layers.Layer):
    def __init__(self):
        super(ContextEmbeddingBlock, self).__init__()
        self.gap = K.layers.GlobalAvgPool2D(keepdims=True)
        self.bn = K.layers.BatchNormalization()

    def build(self, input_shape):
        self.conv1x1 = ConvBlock(input_shape[-1], 1, activation='relu')
        self.conv_final = Conv2D(input_shape[-1], 3, activation='relu', padding='same')

    def call(self, inputs, *args, **kwargs):
        x2_gap = self.gap(inputs)
        x2_gap = self.bn(x2_gap)
        x2_gap = self.conv1x1(x2_gap)
        x = inputs + x2_gap
        x = self.conv_final(x)
        return x


class GatherExpandBlock(K.layers.Layer):
    def __init__(self, stride=1, out_channels=128):
        super(GatherExpandBlock, self).__init__()
        self.stride = stride
        self.out_channels = out_channels
        self.conv3x3_base_left = ConvBlock(out_channels, 3, activation='relu', padding='same')
        self.conv1x1_base_left = ConvBlock(out_channels, 1, activation='linear', padding='same')
        if self.stride > 1:
            self.dwconv3x3_stride_left = DWConvBlock(6, 3, 2, padding='same')
            self.dwconv3x3_stride_right = DWConvBlock(6, 3, 2, padding='same')

            self.dwconv3x3_base_left = DWConvBlock(1, 3, padding='same')
        else:
            self.dwconv3x3_base_left = DWConvBlock(6, 3, padding='same')

    def build(self, input_shape):
        if self.stride > 1:
            self.conv1x1_stride_right = ConvBlock(self.out_channels, 1, activation='linear', padding='same')
            self.conv1x1_stride_left = ConvBlock(self.out_channels, 1, activation='linear', padding='same')
        else:
            self.conv1x1_base_left = ConvBlock(input_shape[-1], 1, activation='linear', padding='same')

    def call(self, inputs, *args, **kwargs):
        if self.stride > 1:
            x_left = self.conv3x3_base_left(inputs)
            x_left = self.dwconv3x3_stride_left(x_left)
            x_left = self.dwconv3x3_base_left(x_left)
            x_left = self.conv1x1_base_left(x_left)
            x_right = self.dwconv3x3_stride_right(inputs)
            x_right = self.conv1x1_stride_right(x_right)
            return tf.nn.relu(x_left + x_right)
        else:
            x = self.conv3x3_base_left(inputs)
            x = self.dwconv3x3_base_left(x)
            x = self.conv1x1_base_left(x)
            return tf.nn.relu(inputs + x)


class Aggregator(K.layers.Layer):
    def __init__(self):
        super(Aggregator, self).__init__()
        self.bn = K.layers.BatchNormalization()
        self.avg_pool = K.layers.AveragePooling2D(3, 2, padding='same')
        # ------ Detail ------- #
        self.detail_dwconv_a = DWConvBlock(1, 3, padding='same')
        # ------ Semantic ------- #
        self.semantic_dwconv_b = DWConvBlock(1, 3, padding='same')

    def build(self, input_shape):
        # ------ Detail ------- #
        self.detail_1x1conv_a = Conv2D(input_shape[0][-1], 1)
        self.detail_3x3conv_b = ConvBlock(input_shape[0][-1], kernel_size=3, strides=2, padding='same')
        # ------ Semantic ------- #
        self.semantic_1x1conv_b = Conv2D(input_shape[1][-1], 1)
        self.semantic_3x3conv_a = ConvBlock(input_shape[1][-1], kernel_size=3, padding='same')

        self.final_conv = ConvBlock(input_shape[0][-1], 3, padding='same')

    def call(self, inputs, *args, **kwargs):
        detail, semantic = inputs

        x_a = self.detail_dwconv_a(detail)
        x_a = self.detail_1x1conv_a(x_a)
        x_b = self.detail_3x3conv_b(detail)
        x_b = self.avg_pool(x_b)

        y_a = self.semantic_3x3conv_a(semantic)
        y_a = tf.nn.sigmoid(tf.image.resize(y_a, tf.shape(detail)[1:3]))
        y_b = self.semantic_dwconv_b(semantic)
        y_b = tf.nn.sigmoid(self.semantic_1x1conv_b(y_b))

        a = y_a * x_a
        b = tf.image.resize(y_b * x_b, tf.shape(a)[1:3])
        return self.final_conv(a + b)


class BiSeNetv2(K.Model):
    def __init__(self, classes=21, activation='relu', backbone='resnet50', alpha=1.0, seg_channels=64):
        super(BiSeNetv2, self).__init__()
        self.backbone = backbone
        # =========== Detail Branch =========== #
        self.detail_convblock1 = K.Sequential([
            ConvBlock(64, 3, 2, padding='same', activation=activation),
            ConvBlock(64, 3, 1, padding='same', activation=activation)
        ])
        self.detail_convblock2 = K.Sequential([
            ConvBlock(64, 3, 2, padding='same', activation=activation),
            ConvBlock(64, 3, 1, padding='same', activation=activation),
            ConvBlock(64, 3, 1, padding='same', activation=activation)
        ])
        self.detail_convblock3 = K.Sequential([
            ConvBlock(128, 3, 2, padding='same', activation=activation),
            ConvBlock(128, 3, 1, padding='same', activation=activation),
            ConvBlock(128, 3, 1, padding='same', activation=activation)
        ])
        # =========== Semantic Branch =========== #
        self.stem = StemBlock(int(16 * alpha))
        self.ge1 = K.Sequential([
            GatherExpandBlock(stride=2, out_channels=int(32 * alpha)),
            GatherExpandBlock(out_channels=int(32 * alpha))
        ])
        self.ge2 = K.Sequential([
            GatherExpandBlock(stride=2, out_channels=int(64 * alpha)),
            GatherExpandBlock(out_channels=int(64 * alpha))
        ])
        self.ge3 = K.Sequential([
            GatherExpandBlock(stride=2, out_channels=int(128 * alpha)),
            GatherExpandBlock(out_channels=int(128 * alpha)),
            GatherExpandBlock(out_channels=int(128 * alpha)),
            GatherExpandBlock(out_channels=int(128 * alpha))
        ])
        self.ce = ContextEmbeddingBlock()
        # =========== Segmentation Head =========== #
        self.seg_head = K.Sequential([ConvBlock(seg_channels, 3, padding='same'), K.layers.Conv2D(classes, 1, padding='same')])
        # ========== Aggregation Head ============ #
        self.aggregator = Aggregator()

    def call(self, inputs, training=None, mask=None):
        original_size = tf.shape(inputs)[1:3]
        # ========= Detail ============ #
        x1_s1 = self.detail_convblock1(inputs)         # Stride /2
        x1_s2 = self.detail_convblock2(x1_s1)             # Stride /4
        x1_s3 = self.detail_convblock3(x1_s2)             # Stride /8
        # ========= Semantic ========== #
        x2_s2 = self.stem(inputs)                   # Stride /4
        x2_s3 = self.ge1(x2_s2)                     # Stride /8
        x2_s4 = self.ge2(x2_s3)                     # Stride /16
        x2_s5 = self.ge3(x2_s4)                     # Stride /32
        x2_ce = self.ce(x2_s5)

        final_feat = self.seg_head(tf.image.resize(self.aggregator((x1_s3, x2_ce)), original_size))
        if training:
            out_s3 = tf.image.resize(self.seg_head(x2_s3), original_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            out_s4 = tf.image.resize(self.seg_head(x2_s4), original_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            out_s5 = tf.image.resize(self.seg_head(x2_s5), original_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            return final_feat, out_s3, out_s4, out_s5
        return final_feat

if __name__ == "__main__":
    x = tf.random.normal((1, 512, 1024, 3))
    bisenet = BiSeNetv2(classes=19)
    z = bisenet(x)
    print(z.shape)