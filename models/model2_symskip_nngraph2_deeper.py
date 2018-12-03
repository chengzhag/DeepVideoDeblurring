import tensorflow as tf

def create_model(input, training):
    def convBNrelu(input, filters, kernel_size, strides,
                   kernel_initializer, bias_initializer, training, relu=True):
        convOut = tf.layers.conv2d(
            input,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="SAME",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer
        )
        bnOut = tf.layers.batch_normalization(
            convOut,
            # momentum=0.1,  # torch default
            # epsilon=1e-3,
            training=training
        )
        if relu:
            reluOut = tf.nn.relu(bnOut)
            return reluOut
        else:
            return bnOut

    def upConvBNaddRelu(input, inputAdd, filters, kernel_size, strides,
                        kernel_initializer, bias_initializer, training):
        upConvOut = tf.layers.conv2d_transpose(
            input,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="SAME",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer
        )
        bnOut = tf.layers.batch_normalization(
            upConvOut,
            # momentum=0.1,  # torch default
            # epsilon=1e-3,
            training=training
        )
        reluOut = tf.nn.relu(bnOut + inputAdd)
        return reluOut

    kernelInitializer = tf.variance_scaling_initializer
    biasInitializer = tf.variance_scaling_initializer

    ref = tf.slice(input, [0, 0, 0, 6], [-1, -1, -1, 3])
    F0 = convBNrelu(input, 64, (5, 5), (1, 1), kernelInitializer, biasInitializer, training)

    D1 = convBNrelu(F0, 64, (3, 3), (2, 2), kernelInitializer, biasInitializer, training)
    F1 = convBNrelu(D1, 128, (3, 3), (1, 1), kernelInitializer, biasInitializer, training)
    F2 = convBNrelu(F1, 128, (3, 3), (1, 1), kernelInitializer, biasInitializer, training)

    D2 = convBNrelu(F2, 256, (3, 3), (2, 2), kernelInitializer, biasInitializer, training)
    F3 = convBNrelu(D2, 256, (3, 3), (1, 1), kernelInitializer, biasInitializer, training)
    F4 = convBNrelu(F3, 256, (3, 3), (1, 1), kernelInitializer, biasInitializer, training)
    F5 = convBNrelu(F4, 256, (3, 3), (1, 1), kernelInitializer, biasInitializer, training)

    D3 = convBNrelu(F5, 512, (3, 3), (2, 2), kernelInitializer, biasInitializer, training)
    F6 = convBNrelu(D3, 512, (3, 3), (1, 1), kernelInitializer, biasInitializer, training)
    F7 = convBNrelu(F6, 512, (3, 3), (1, 1), kernelInitializer, biasInitializer, training)
    F8 = convBNrelu(F7, 512, (3, 3), (1, 1), kernelInitializer, biasInitializer, training)

    S1 = upConvBNaddRelu(F8, F5, 256, (4, 4), (2, 2), kernelInitializer, biasInitializer, training)
    F9 = convBNrelu(S1, 256, (3, 3), (1, 1), kernelInitializer, biasInitializer, training)
    F10 = convBNrelu(F9, 256, (3, 3), (1, 1), kernelInitializer, biasInitializer, training)
    F11 = convBNrelu(F10, 256, (3, 3), (1, 1), kernelInitializer, biasInitializer, training)

    S2 = upConvBNaddRelu(F11, F2, 128, (4, 4), (2, 2), kernelInitializer, biasInitializer, training)
    F12 = convBNrelu(S2, 128, (3, 3), (1, 1), kernelInitializer, biasInitializer, training)
    F13 = convBNrelu(F12, 64, (3, 3), (1, 1), kernelInitializer, biasInitializer, training)

    S3 = upConvBNaddRelu(F13, F0, 64, (4, 4), (2, 2), kernelInitializer, biasInitializer, training)
    F14 = convBNrelu(S3, 15, (3, 3), (1, 1), kernelInitializer, biasInitializer, training)
    F15 = convBNrelu(F14, 3, (3, 3), (1, 1), kernelInitializer, biasInitializer, training, False)

    S4 = tf.nn.sigmoid(ref + F15)

    return S4