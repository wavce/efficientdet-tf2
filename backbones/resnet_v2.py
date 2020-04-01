import tensorflow as tf
from backbones.backbone import Backbone
from core.layers import build_convolution
from core.layers import build_normalization
from backbones.resnet_common import bottleneck_v2


class ResNetV2(Backbone):
    BLOCKS = {
        "resnet50": [3, 4, 6, 3],
        "resnet101": [3, 4, 23, 3],
        "resnet152": [3, 8, 36, 3]
    } 

    def __init__(self,
                 name,
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(-1, ),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 weight_decay=0.,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(ResNetV2, self).__init__(name,
                                       convolution=convolution,
                                       normalization=normalization,
                                       activation=activation,
                                       output_indices=output_indices,
                                       strides=strides,
                                       dilation_rates=dilation_rates,
                                       frozen_stages=frozen_stages,
                                       weight_decay=weight_decay,
                                       dropblock=dropblock)

        # self._rgb_mean = tf.constant([123.68, 116.78, 103.94], dtype=self.dtype)
        # self._rgb_mean = tf.constant([0.485, 0.456, 0.406], dtype=self.dtype)
        # self._rgb_std = tf.constant([0.229, 0.224, 0.225], dtype=self.dtype)

        self.normalization = normalization
        self.blocks = ResNetV2.BLOCKS[name]
        self.num_classes = num_classes
        self.drop_rate = drop_rate

        self.model = self.resnet_v2()

    def stack(self, x, filters, strides, dilation_rate, trainable, blocks, name=None):
        x, preact = bottleneck_v2(x,
                                  convolution=self.convolution,
                                  filters=filters,
                                  strides=1,
                                  dilation_rate=dilation_rate,
                                  normalization=self.normalization,
                                  activation=self.activation,
                                  trainable=trainable,
                                  weight_decay=self.weight_decay,
                                  dropblock=self.dropblock,
                                  use_conv_shortcut=True,
                                  name=name + '_block1')
        for i in range(2, blocks):
            x, _ = bottleneck_v2(x,
                                 convolution=self.convolution,
                                 filters=filters,
                                 strides=1,
                                 dilation_rate=dilation_rate,
                                 normalization=self.normalization,
                                 trainable=trainable,
                                 weight_decay=self.weight_decay,
                                 dropblock=self.dropblock,
                                 use_conv_shortcut=False,
                                 name=name + '_block' + str(i))
        x, _ = bottleneck_v2(x,
                             convolution=self.convolution,
                             filters=filters,
                             strides=strides,
                             dilation_rate=1 if strides > 1 else dilation_rate,
                             normalization=self.normalization,
                             trainable=trainable,
                             weight_decay=self.weight_decay,
                             dropblock=self.dropblock,
                             use_conv_shortcut=False,
                             name=name + '_block' + str(blocks))

        return x, preact

    def resnet_v2(self):
        trainable = 0 not in self.frozen_stages
        
        x = tf.keras.layers.Lambda(function=lambda inp: inp / 127.5 - 1.,
                                   name="mean_subtraction")(self.img_input)
        x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(x)
        x1 = build_convolution(self.convolution,
                               filters=64,
                               kernel_size=7,
                               strides=self.strides[0],
                               padding="valid",
                               dilation_rate=self.dilation_rates[0],
                               trainable=trainable,
                               kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                               use_bias=True,
                               name="conv1_conv")(x)

        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")(x1)
        x = tf.keras.layers.MaxPool2D((3, 3), strides=self.strides[1], padding="valid", name="pool1_pool")(x)

        trainable = 1 not in self.frozen_stages
        x, _ = self.stack(x, 64, self.strides[2], self.dilation_rates[1], trainable, self.blocks[0], "conv2")

        trainable = 2 not in self.frozen_stages
        x, preact2 = self.stack(x, 128, self.strides[3], self.dilation_rates[2], trainable, self.blocks[1], "conv3")

        trainable = 3 not in self.frozen_stages
        x, preact3 = self.stack(x, 256, self.strides[4], self.dilation_rates[3], trainable, self.blocks[2], "conv4")

        trainable = 4 not in self.frozen_stages
        x, preact4 = self.stack(x, 512, 1, self.dilation_rates[4], trainable, self.blocks[3], "conv5")
        x = build_normalization(**self.normalization, name="post_bn")(x)
        x5 = tf.keras.layers.Activation(**self.activation, name="post_relu")(x)

        if self._is_classifier:
            x = tf.keras.layers.GlobalAvgPool2D(name="avg_pool")(x5)
            x = tf.keras.layers.Dropout(rate=self.drop_rate)(x)
            x = tf.keras.layers.Dense(self.num_classes, activation="softmax", name="probs")(x)

            return tf.keras.Model(inputs=self.img_input, outputs=x, name="resnet" + str(self.depth) + "v2")

        outputs = [o for i, o in enumerate([x1, preact2, preact3, preact4, x5]) if i in self.output_indices]

        return outputs


if __name__ == '__main__':
    resnet = ResNetV2(50)
    resnet.init_weights("/home/bail/Workspace/pretrained_weights/resnet50v2.h5")
    # resnet.init_weights("/home/bail/workspace/pretrained_weights/resnet_v2_50.ckpt")

    with tf.io.gfile.GFile("/home/bail/Documents/pandas.jpg", "rb") as gf:
        images = tf.image.decode_jpeg(gf.read())

    images = tf.image.resize(images, (224, 224))
    images = tf.expand_dims(images, axis=0)
    cls = resnet(images, training=False)

    print(tf.argmax(tf.squeeze(cls, axis=0)))
    print(tf.reduce_max(cls))

