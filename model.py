import keras


def conv_layer(num_filters, kernel_size, in_feature):
    conv = keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size,
                               padding='same',
                               kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(5e-4))(in_feature)
    bn = keras.layers.BatchNormalization()(conv)
    out_feature = keras.layers.Activation('relu')(bn)
    return out_feature


def conv_block(num_filters, kernel_size, in_feature):
    conv1 = conv_layer(num_filters, kernel_size, in_feature)
    conv2 = conv_layer(num_filters, kernel_size, conv1)
    return conv2


def residual_conv_block(num_filters, kernel_size, in_feature):
    conv1 = conv_layer(num_filters, kernel_size, in_feature)
    conv2 = conv_layer(num_filters, kernel_size, conv1)
    skip = conv_layer(num_filters, kernel_size, in_feature)
    out_feature = keras.layers.Add()([skip, conv2])
    return out_feature


def upsample_block(num_filters, kernel_size, in_feature, skip):
    deconv = keras.layers.Conv2DTranspose(filters=num_filters, kernel_size=kernel_size,
                                          padding='same', strides=2,
                                          kernel_initializer='he_normal',
                                          kernel_regularizer=keras.regularizers.l2(5e-4))(in_feature)
    merge = keras.layers.concatenate([deconv, skip], axis=-1)
    return merge


def attention_block(num_filters, in_feature, skip):
    g1 = conv_layer(num_filters=num_filters, kernel_size=1, in_feature=skip)
    x1 = conv_layer(num_filters=num_filters, kernel_size=1, in_feature=in_feature)

    g1_x1 = keras.layers.Add()([g1, x1])
    psi = keras.layers.Conv2D(1, kernel_size=1)(g1_x1)
    psi = keras.layers.BatchNormalization()(psi)
    psi = keras.layers.Activation('sigmoid')(psi)
    out_feature = keras.layers.Multiply()([in_feature, psi])
    return out_feature


def attention_upsample_block(num_filters, kernel_size, in_feature, skip):
    deconv = keras.layers.Conv2DTranspose(filters=num_filters, kernel_size=kernel_size,
                                          padding='same',
                                          kernel_initializer='he_normal',
                                          kernel_regularizer=keras.regularizers.l2(5e-4))(in_feature)
    merge = attention_block(num_filters, deconv, skip)
    return merge


def dilated_conv_layer(num_filters, kernel_size, dilation_rate, in_feature):
    conv = keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size,
                               padding='same',
                               dilation_rate=dilation_rate,
                               kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(5e-4))(in_feature)
    bn = keras.layers.BatchNormalization()(conv)
    out_feature = keras.layers.Activation('relu')(bn)
    return out_feature


def unet(num_filters=32, kernel_size=3, input_shape=(224, 224, 3), num_classes=1, pretrain_path=None,
         attention=True, residual=True, dilation=True):
    if residual:
        cnn = residual_conv_block
    else:
        cnn = conv_block
    if attention:
        upsample = attention_upsample_block
    else:
        upsample = upsample_block

    inputs = keras.layers.Input(input_shape)
    conv0 = cnn(num_filters, kernel_size=5, in_feature=inputs)
    block1 = cnn(num_filters, kernel_size, conv0)
    pool1 = keras.layers.MaxPooling2D()(block1)
    block2 = cnn(num_filters * 2, kernel_size, pool1)
    pool2 = keras.layers.MaxPooling2D()(block2)
    block3 = cnn(num_filters * 4, kernel_size, pool2)
    pool3 = keras.layers.MaxPooling2D()(block3)
    block4 = cnn(num_filters * 8, kernel_size, pool3)
    pool4 = keras.layers.MaxPooling2D()(block4)
    if dilation:
        bottleneck = dilated_conv_layer(num_filters * 16, kernel_size, pool4, 2)
        bottleneck = dilated_conv_layer(num_filters * 16, kernel_size, bottleneck, 3)
        bottleneck = dilated_conv_layer(num_filters * 16, kernel_size, bottleneck, 5)
    else:
        bottleneck = conv_layer(num_filters * 16, kernel_size, pool4)
        bottleneck = conv_layer(num_filters * 16, kernel_size, bottleneck)
        bottleneck = conv_layer(num_filters * 16, kernel_size, bottleneck)
    upsample5 = upsample(num_filters * 8, kernel_size, bottleneck, block4)
    block5 = cnn(num_filters * 8, kernel_size, upsample5)
    upsample6 = upsample(num_filters * 4, kernel_size, block5, block3)
    block6 = cnn(num_filters * 4, kernel_size, upsample6)
    upsample7 = upsample(num_filters * 2, kernel_size, block6, block2)
    block7 = cnn(num_filters * 2, kernel_size, upsample7)
    upsample8 = upsample(num_filters, kernel_size, block7, block1)
    block8 = cnn(num_filters, kernel_size, upsample8)
    out = keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(block8)
    model = keras.models.Model(input=inputs, output=out)

    if pretrain_path:
        model.load_weights(pretrain_path)

    return model
