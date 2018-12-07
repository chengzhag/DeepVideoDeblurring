import tensorflow as tf
import tensorflow.contrib.slim as slim


def create_model(input, training):
    with tf.name_scope('ref'):
        ref = tf.slice(input, [0, 0, 0, 6], [-1, -1, -1, 3])

    with slim.arg_scope([slim.conv2d],
                        kernel_size=(3, 3),
                        stride=1,
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm):
        with slim.arg_scope([slim.conv2d_transpose],
                            kernel_size=(4, 4),
                            stride=2,
                            activation_fn=None,
                            normalizer_fn=slim.batch_norm):
            with slim.arg_scope([slim.batch_norm],
                                decay=0.99,
                                is_training=training):
                F0 = slim.conv2d(input, 64, kernel_size=(5, 5), scope='F0')

                D1 = slim.conv2d(F0, 64, stride=2, scope='D1')
                F1 = slim.conv2d(D1, 128, scope='F1')
                F2 = slim.conv2d(F1, 128, scope='F2')

                D2 = slim.conv2d(F2, 256, stride=2, scope='D2')
                F3 = slim.conv2d(D2, 256, scope='F3')
                F4 = slim.conv2d(F3, 256, scope='F4')
                F5 = slim.conv2d(F4, 256, scope='F5')

                D3 = slim.conv2d(F5, 512, stride=2, scope='D3')
                F6 = slim.conv2d(D3, 512, scope='F6')
                F7 = slim.conv2d(F6, 512, scope='F7')
                F8 = slim.conv2d(F7, 512, scope='F8')

                U1 = slim.conv2d_transpose(F8, 256, scope='U1')
                with tf.name_scope('S1'): S1 = tf.nn.relu(U1 + F5)
                F9 = slim.conv2d(S1, 256, scope='F9')
                F10 = slim.conv2d(F9, 256, scope='F10')
                F11 = slim.conv2d(F10, 256, scope='F11')

                U2 = slim.conv2d_transpose(F11, 128, scope='U2')
                with tf.name_scope('S2'): S2 = tf.nn.relu(U2 + F2)
                F12 = slim.conv2d(S2, 128, scope='F12')
                F13 = slim.conv2d(F12, 64, scope='F13')

                U3 = slim.conv2d_transpose(F13, 64, scope='U3')
                with tf.name_scope('S3'): S3 = tf.nn.relu(U3 + F0)
                F14 = slim.conv2d(S3, 15, scope='F14')
                F15 = slim.conv2d(F14, 3, scope='F15', activation_fn=None)

    with tf.name_scope('S4'):
        S4 = tf.nn.sigmoid(ref + F15)

    return S4
