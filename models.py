import tensorflow as  tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


class ZeroPadding(layers.Layer):
    def __init__(self, values):
        super(ZeroPadding, self).__init__()
        self.values = values

    def call(self, x):
        x = tf.pad(x, [[0, 0], [0, 0], [self.values[0], self.values[1]]], mode='CONSTANT', constant_values=0)
        return x


class ConvBlock(layers.Layer):
    def __init__(self, filters=256, kernel_size=3, padding='same', pool=False):
        super(ConvBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.pool = pool

        self.conv1 = layers.Conv1D(self.filters, self.kernel_size, strides=1, padding=self.padding)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')

        self.conv2 = layers.Conv1D(self.filters, self.kernel_size, strides=1, padding=self.padding)
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.Activation('relu')

        self.shortcut_conv = layers.Conv1D(self.filters, self.kernel_size, strides=2, padding=self.padding)
        self.shortcut_bn = layers.BatchNormalization()
        self.shortcut_pad = layers.MaxPooling1D(pool_size=kernel_size, strides=2, padding='same')
        self.shortcut_add = layers.Add()
        self.zero_padding = ZeroPadding([int(self.filters // 2), self.filters - int(self.filters // 2)])

    def call(self, inputs):
        cnn1 = self.conv1(inputs)
        cnn1 = self.bn1(cnn1)
        cnn1 = self.relu1(cnn1)

        cnn2 = self.conv2(cnn1)
        cnn2 = self.bn2(cnn2)
        cnn2 = self.relu2(cnn2)

        if self.pool:
            downsample = self.shortcut_conv(cnn2)
            downsample = self.shortcut_bn(downsample)
            conv_pool = self.shortcut_pad(cnn2)
            conv_shortcut = self.shortcut_add([downsample, conv_pool])
            conv_project = self.zero_padding(conv_shortcut)
            return conv_project

        else:
            conv_shortcut = self.shortcut_add([cnn2, inputs])
            return conv_shortcut


class K_Max_Pooling(layers.Layer):
    def __init__(self, k):
        super(K_Max_Pooling, self).__init__()
        self.k = k

    def call(self, inputs):
        input_transpose = layers.Permute((2, 1))(inputs)
        top_k, _ = tf.math.top_k(input_transpose, k=self.k, sorted=False)
        top_k = layers.Permute((2, 1))(top_k)
        return top_k


def VDCNN(self, depth):
    model_depth = {9: [1, 1, 1, 1], 17: [2, 2, 2, 2], 29: [5, 5, 2, 2], 49: [8, 8, 5, 3]}

    inputs = layers.Input((self.max_len,))
    embedding = layers.Embedding(self.vocab_size, self.embedding_size)(inputs)

    temp_conv_64 = layers.Conv1D(filters=64, kernel_size=self.kernel_size, strides=1, padding='same')(embedding)
    for i in range(model_depth[depth][0] - 1):  # 64
        temp_conv_64 = ConvBlock(filters=64, kernel_size=self.kernel_size)(temp_conv_64)
    temp_conv_128 = ConvBlock(filters=64, kernel_size=3, pool=True)(temp_conv_64)

    for i in range(model_depth[depth][1] - 1):  # 128
        temp_conv_128 = ConvBlock(filters=128, kernel_size=self.kernel_size)(temp_conv_128)
    temp_conv_256 = ConvBlock(filters=128, kernel_size=self.kernel_size, pool=True)(temp_conv_128)

    for i in range(model_depth[depth][2] - 1):  # 256
        temp_conv_256 = ConvBlock(filters=256, kernel_size=self.kernel_size)(temp_conv_256)
    temp_conv_512 = ConvBlock(filters=256, kernel_size=self.kernel_size, pool=True)(temp_conv_256)

    for i in range(model_depth[depth][3] - 1):  # 512
        temp_conv_512 = ConvBlock(filters=512, kernel_size=self.kernel_size)(temp_conv_512)
    temp_conv_512 = ConvBlock(filters=512, kernel_size=self.kernel_size, pool=True)(temp_conv_512)

    output = K_Max_Pooling(k=self.k)(temp_conv_512)

    output = layers.Flatten()(output)

    output = layers.Dense(2048, activation='relu')(output)

    output = layers.Dense(2048, activation='relu')(output)

    output = layers.Dense(self.class_num, activation='softmax')(output)

    model = Model(inputs=inputs, outputs=output)

    model.compile(loss=self.loss, optimizer=self.opt, metrics=['accuracy'])

    model.summary()

    return model
