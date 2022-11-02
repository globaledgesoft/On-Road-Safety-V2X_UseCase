#import tensorflow as tf
#import tensorflow.keras.backend as K

WEIGHT_DECAY = 0.  # 5e-4
LEAKY_ALPHA = 0.1

def mish(x):
    #return x * K.tanh(K.softplus(x))
    return x * K.tanh(K.log(1+K.exp(x)))

def myConv2D(*args, **kwargs):
    my_conv_kwargs = {"kernel_regularizer": tf.keras.regularizers.l2(WEIGHT_DECAY),
                           "kernel_initializer": tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
                           "padding": "valid" if kwargs.get(
                               "strides") == (2, 2) else "same"}
    my_conv_kwargs.update(kwargs)

    return tf.keras.layers.Conv2D(*args, **my_conv_kwargs)


def myConv2D_BN_Leaky(*args, **kwargs):
    without_bias_kwargs = {"use_bias": False}
    without_bias_kwargs.update(kwargs)

    def wrapper(x):
        x = myConv2D(*args, **without_bias_kwargs)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA)(x)
        return x

    return wrapper


def myConv2D_BN_Mish(*args, **kwargs):
    """my Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)

    def wrapper(x):
        x = myConv2D(*args, **no_bias_kwargs)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(mish)(x)
        return x

    return wrapper


def myBlock(num_filters, niter, all_narrow=True):
    '''A series of resblocks starting with a downsampling Convolution2D'''

    # my uses left and top padding instead of 'same' mode
    def wrapper(x):
        x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        x = myConv2D_BN_Mish(num_filters, (3, 3), strides=(2, 2))(x)
        shortcut = myConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(x)
        x = myConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(x)
        for _ in range(niter):
            y = myConv2D_BN_Mish(num_filters // 2, (1, 1))(x)
            y = myConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (3, 3))(y)

            x = tf.keras.layers.Add()([x, y])
        x = myConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(x)
        x = tf.keras.layers.Concatenate()([x, shortcut])
        x = myConv2D_BN_Mish(num_filters, (1, 1))(x)
        return x

    return wrapper


def my_model(iou_threshold, score_threshold, max_outputs, num_classes, strides, mask, anchors,
                input_size=None,
                name=None):

    if input_size is None:
        x = inputs = tf.keras.Input([None, None, 3])
    else:
        x = inputs = tf.keras.Input([input_size, input_size, 3])

    x = myConv2D_BN_Mish(32, (3, 3))(x)
    x = myBlock(64, 1, False)(x)
    x = myBlock(128, 2)(x)
    x = x_131 = myBlock(256, 8)(x)
    x = x_204 = myBlock(512, 8)(x)
    x = myBlock(1024, 4)(x)

    # 19x19 head
    x = myConv2D_BN_Leaky(512, (1, 1))(x)
    x = myConv2D_BN_Leaky(1024, (3, 3))(x)
    x = myConv2D_BN_Leaky(512, (1, 1))(x)

    maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x)
    maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
    maxpool3 = tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)

    x = tf.keras.layers.Concatenate()([maxpool1, maxpool2, maxpool3, x])
    x = myConv2D_BN_Leaky(512, (1, 1))(x)
    x = myConv2D_BN_Leaky(1024, (3, 3))(x)
    x = x_19 = myConv2D_BN_Leaky(512, (1, 1))(x)

    x = myConv2D_BN_Leaky(256, (1, 1))(x)
    x19_upsample = tf.keras.layers.UpSampling2D(2)(x)

    # 38x38 head
    x = myConv2D_BN_Leaky(256, (1, 1))(x_204)
    x = tf.keras.layers.Concatenate()([x, x19_upsample])
    x = myConv2D_BN_Leaky(256, (1, 1))(x)
    x = myConv2D_BN_Leaky(512, (3, 3))(x)
    x = myConv2D_BN_Leaky(256, (1, 1))(x)
    x = myConv2D_BN_Leaky(512, (3, 3))(x)
    x = x_38 = myConv2D_BN_Leaky(256, (1, 1))(x)

    x = myConv2D_BN_Leaky(128, (1, 1))(x)
    x38_upsample = tf.keras.layers.UpSampling2D(2)(x)

    # 76x76 head
    x = myConv2D_BN_Leaky(128, (1, 1))(x_131)
    x = tf.keras.layers.Concatenate()([x, x38_upsample])
    x = myConv2D_BN_Leaky(128, (1, 1))(x)
    x = myConv2D_BN_Leaky(256, (3, 3))(x)
    x = myConv2D_BN_Leaky(128, (1, 1))(x)
    x = myConv2D_BN_Leaky(256, (3, 3))(x)
    x = x_76 = myConv2D_BN_Leaky(128, (1, 1))(x)

    # 76x76 output
    x = myConv2D_BN_Leaky(256, (3, 3))(x)
    output_2 = myConv2D(len(mask[2]) * (num_classes + 5), (1, 1))(x)

    # 38x38 output
    x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x_76)
    x = myConv2D_BN_Leaky(256, (3, 3), strides=(2, 2))(x)
    x = tf.keras.layers.Concatenate()([x, x_38])
    x = myConv2D_BN_Leaky(256, (1, 1))(x)
    x = myConv2D_BN_Leaky(512, (3, 3))(x)
    x = myConv2D_BN_Leaky(256, (1, 1))(x)
    x = myConv2D_BN_Leaky(512, (3, 3))(x)
    x = x_38 = myConv2D_BN_Leaky(256, (1, 1))(x)

    x = myConv2D_BN_Leaky(512, (3, 3))(x)
    output_1 = myConv2D(len(mask[2]) * (num_classes + 5), (1, 1))(x)

    # 19x19 output
    x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x_38)
    x = myConv2D_BN_Leaky(512, (3, 3), strides=(2, 2))(x)
    x = tf.keras.layers.Concatenate()([x, x_19])
    x = myConv2D_BN_Leaky(512, (1, 1))(x)
    x = myConv2D_BN_Leaky(1024, (3, 3))(x)
    x = myConv2D_BN_Leaky(512, (1, 1))(x)
    x = myConv2D_BN_Leaky(1024, (3, 3))(x)
    x = myConv2D_BN_Leaky(512, (1, 1))(x)

    x = myConv2D_BN_Leaky(1024, (3, 3))(x)
    output_0 = myConv2D(len(mask[2]) * (num_classes + 5), (1, 1))(x)

    model = tf.keras.Model(inputs, [output_0, output_1, output_2], name=name)

    return model




