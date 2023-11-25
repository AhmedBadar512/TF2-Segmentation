import tensorflow_addons as tfa
import tensorflow as tf
import keras as K


class ConvBlock(K.layers.Layer):
    """ ConBlock layer consists of Conv2D + Normalization + Activation.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='same',
                 use_bias=False,
                 norm_layer=None,
                 activation='linear',
                 dilation_rate=(1, 1),
                 **kwargs):
        super(ConvBlock, self).__init__()
        self.conv2d = K.layers.Conv2D(filters,
                                      kernel_size,
                                      strides,
                                      padding,
                                      dilation_rate=dilation_rate,
                                      use_bias=use_bias,
                                      kernel_initializer=K.initializers.HeNormal(),
                                      **kwargs)
        self.activation = K.layers.Activation(activation)
        if norm_layer:
            self.normalization = norm_layer()
        else:
            self.normalization = tf.identity

    def call(self, inputs, training=None, **kwargs):
        x = self.conv2d(inputs)
        x = self.normalization(x, training=training)
        x = self.activation(x)

        return x


class CombConvLayer_tf(K.layers.Layer):
    def __init__(self, out_channels, kernels=1, strides=1, **kwargs):
        super(CombConvLayer_tf, self).__init__()
        self.out_channels = out_channels
        self.kernels = kernels
        self.strides = strides
        self.conv_layer = ConvBlock(self.out_channels, kernels, strides, norm_layer=K.layers.BatchNormalization, **kwargs)
        self.dw_conv_layer = K.layers.DepthwiseConv2D(3, padding="same", use_bias=False)
        self.bn = K.layers.BatchNormalization()

    def call(self, inputs, **kwargs):
        x = self.conv_layer(inputs)
        x = self.dw_conv_layer(x)
        return self.bn(x)


class HarDBlock_tf(K.layers.Layer):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        return out_channels, link

    def __init__(self, growth_rate, grmul, n_layers, keepbase=False, dwconv=False):
        super(HarDBlock_tf, self).__init__()
        self.keepbase = keepbase
        self.links = []
        self.out_channels = 0  # if upsample else in_channels
        self.n_layers = n_layers
        self.growth_rate = growth_rate
        self.grmul = grmul
        self.dwconv = dwconv
        self.layers = []

    def build(self, input_shape):
        layers_ = []
        for i in range(self.n_layers):
            outch, link = self.get_link(i + 1, input_shape[-1], self.growth_rate, self.grmul)
            self.links.append(link)
            if self.dwconv:
                layers_.append(CombConvLayer_tf(outch))
            else:
                layers_.append(ConvBlock(outch, 3, 1, activation="relu", norm_layer=K.layers.BatchNormalization))

            if (i % 2 == 0) or (i == self.n_layers - 1):
                self.out_channels += outch
        self.layers = layers_

    def call(self, inputs, *args, **kwargs):
        layers_ = [inputs]
        for l in range(len(self.layers)):
            link = self.links[l]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = tf.concat(tin, axis=-1)
            else:
                x = tin[0]
            out = self.layers[l](x)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepbase) or (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        return tf.concat(out_, axis=-1)


class ExtraLayers(K.layers.Layer):
    def __init__(self):
        super(ExtraLayers, self).__init__()
        self.convbnrelu1_1 = ConvBlock(256, kernel_size=1, strides=1, norm_layer=K.layers.BatchNormalization)
        self.convbnrelu1_2 = ConvBlock(512, kernel_size=3, strides=2, norm_layer=K.layers.BatchNormalization)
        self.convbnrelu2_1 = ConvBlock(256, kernel_size=1, strides=1, norm_layer=K.layers.BatchNormalization)
        self.convbnrelu2_2 = ConvBlock(512, kernel_size=3, strides=2, norm_layer=K.layers.BatchNormalization)
        self.convbnrelu3_1 = ConvBlock(256, kernel_size=1, strides=1, norm_layer=K.layers.BatchNormalization)
        self.convbnrelu3_2 = ConvBlock(512, kernel_size=3, strides=2, norm_layer=K.layers.BatchNormalization)
        self.avgpool = tfa.layers.AdaptiveAveragePooling2D(1)

    def call(self, inputs, *args, **kwargs):
        feature_maps = list()
        x = self.convbnrelu1_1(inputs)
        x = self.convbnrelu1_2(x)
        feature_maps.append(x)
        x = self.convbnrelu2_1(x)
        x = self.convbnrelu2_2(x)
        feature_maps.append(x)
        # x = self.convbnrelu3_1(x)
        # x = self.convbnrelu3_2(x)
        # feature_maps.append(x)
        x = self.avgpool(x)
        feature_maps.append(x)
        return feature_maps


class HarDNet(K.Model):
    def __init__(self, depth_wise=False, arch=85, enable_extra=False,):
        super(HarDNet, self).__init__()
        first_ch = [32, 64]
        second_kernel = 3
        max_pool = True
        grmul = 1.7

        # HarDNet68
        ch_list = [128, 256, 320, 640, 1024]
        gr = [14, 16, 20, 40, 160]
        n_layers = [8, 16, 16, 16, 4]
        downSamp = [1, 0, 1, 1, 0]

        if arch == 85:
            # HarDNet85
            first_ch = [48, 96]
            ch_list = [192, 256, 320, 480, 720, 1280]
            gr = [24, 24, 28, 36, 48, 256]
            n_layers = [8, 16, 16, 16, 16, 4]
            downSamp = [1, 0, 1, 0, 1, 0]
            drop_rate = 0.2
        elif arch == 39:
            # HarDNet39
            first_ch = [24, 48]
            ch_list = [96, 320, 640, 1024]
            grmul = 1.6
            gr = [16, 20, 64, 160]
            n_layers = [4, 16, 8, 4]
            downSamp = [1, 1, 1, 0]

        if depth_wise:
            second_kernel = 1
            max_pool = False

        blks = len(n_layers)
        self.base = []
        self.base.append(ConvBlock(first_ch[0], 3, 2, norm_layer=K.layers.BatchNormalization))
        self.base.append(ConvBlock(first_ch[1], second_kernel, 1, norm_layer=K.layers.BatchNormalization))

        if max_pool:
            self.base.append(K.layers.MaxPool2D(3, 2, padding="same"))
        else:
            self.base.append(K.layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False))
            self.base.append(K.layers.BatchNormalization())
        # ch = first_ch[1]

        for i in range(blks):
            blk = HarDBlock_tf(gr[i], grmul, n_layers[i], dwconv=depth_wise)
            self.base.append(blk)
            if i == blks - 1 and arch == 85:
                self.base.append(K.layers.Dropout(0.1))
            self.base.append(ConvBlock(ch_list[i], 1, 1, norm_layer=K.layers.BatchNormalization))
            if downSamp[i] == 1:
                if max_pool:
                    self.base.append(K.layers.MaxPool2D(pool_size=2, strides=2))
                else:
                    self.base.append(K.layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False))
                    self.base.append(K.layers.BatchNormalization())
        if enable_extra:
            self.base.append(ExtraLayers())
        else:
            self.base.append(tfa.layers.AdaptiveAveragePooling2D((1, 1)))
        self.ind_layers = {}

    def call(self, inputs, training=None, mask=None):
        """
        Custom backbones should always return a list of outputs to be used by heads
        HarDNet returns outputs as a list of tensors for each step reduction
        """
        outputs = []
        x = inputs
        old_size = 9e9
        for layer in self.base:
            x = layer(x)
            if isinstance(layer, ConvBlock):
                if layer.conv2d.kernel_size == (1, 1):
                    if x.shape[1] == old_size:
                        outputs.pop()
                    outputs.append(x)
                    old_size = x.shape[1]
            elif isinstance(layer, tfa.layers.AdaptiveAveragePooling2D):
                outputs.append(x)
            elif isinstance(layer, ExtraLayers):
                outputs.extend(x)
        return outputs


if __name__ == "__main__":
    # hblock = HarDBlock_tf(24, 1.7, 8)
    # hblock(tf.random.uniform((8, 512, 512, 64)))
    # print(sum([tf.reduce_prod(var.shape) for var in hblock.trainable_variables]))
    model = HarDNet(arch=39, enable_extra=False)
    in_tensor = K.Input((512, 512, 3))
    a_list = model(in_tensor)
    [print(a.shape) for a in a_list]
    # model.summary()