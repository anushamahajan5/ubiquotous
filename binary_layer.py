import tensorflow as tf
import numpy as np

all_layers = []

def hard_sigmoid(x):
    return tf.clip_by_value((x + 1.) / 2., 0, 1)

def round_through(x):
    rounded = tf.round(x)
    return x + tf.stop_gradient(rounded - x)

def binary_tanh_unit(x):
    return 2. * round_through(hard_sigmoid(x)) - 1.

def binary_sigmoid_unit(x):
    return round_through(hard_sigmoid(x))

def binarization(W, H, binary=True, deterministic=False, stochastic=False, srng=None):
    dim = W.get_shape().as_list()

    if not binary or (deterministic and stochastic):
        Wb = W
    else:
        Wb = H * binary_tanh_unit(W / H)

    return Wb

class Dense_BinaryLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, activation=None, use_bias=True,
                 binary=True, stochastic=True, H=1., W_LR_scale="Glorot",
                 kernel_initializer='glorot_normal',
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, trainable=True, name=None, **kwargs):
        super(Dense_BinaryLayer, self).__init__(name=name, trainable=trainable, **kwargs)
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        self.binary = binary
        self.stochastic = stochastic
        self.H = H
        self.W_LR_scale = W_LR_scale
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        all_layers.append(self)

    def build(self, input_shape):
        num_inputs = np.prod(input_shape[-1])
        num_units = self.output_dim

        if self.H == "Glorot":
            self.H = np.float32(np.sqrt(1.5 / (num_inputs + num_units)))
        self.W_LR_scale = np.float32(1. / np.sqrt(1.5 / (num_inputs + num_units)))

        self.kernel_initializer = tf.random_uniform_initializer(-self.H, self.H)
        self.kernel_constraint = lambda w: tf.clip_by_value(w, -self.H, self.H)

        self.b_kernel = self.add_weight('binary_weight',
                                        shape=(input_shape[-1], self.output_dim),
                                        initializer=tf.random_uniform_initializer(-self.H, self.H),
                                        trainable=False)

        super(Dense_BinaryLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        shape = inputs.get_shape().as_list()

        self.b_kernel = binarization(self.kernel, self.H)

        if len(shape) > 2:
            outputs = tf.tensordot(inputs, self.b_kernel, [[len(shape) - 1], [0]])
            if tf.executing_eagerly():
                output_shape = shape[:-1] + [self.output_dim]
                outputs.set_shape(output_shape)
        else:
            outputs = tf.matmul(inputs, self.b_kernel)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

class Conv2D_BinaryLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_num, kernel_size, strides=(1, 1),
                 padding='valid', activation=None, use_bias=True,
                 binary=True, stochastic=True, H=1., W_LR_scale="Glorot",
                 data_format='channels_last', dilation_rate=(1, 1),
                 kernel_initializer=None, bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, trainable=True, name=None, **kwargs):
        super(Conv2D_BinaryLayer, self).__init__(name=name, trainable=trainable, **kwargs)
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.binary = binary
        self.stochastic = stochastic
        self.H = H
        self.W_LR_scale = W_LR_scale
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        all_layers.append(self)

    def build(self, input_shape):
        num_inputs = np.prod(self.kernel_size) * input_shape[-1]
        num_units = np.prod(self.kernel_size) * self.kernel_num

        if self.H == "Glorot":
            self.H = np.float32(np.sqrt(1.5 / (num_inputs + num_units)))
        self.W_LR_scale = np.float32(1. / np.sqrt(1.5 / (num_inputs + num_units)))

        self.kernel_initializer = tf.random_uniform_initializer(-self.H, self.H)
        self.kernel_constraint = lambda w: tf.clip_by_value(w, -self.H, self.H)

        self.b_kernel = 0

        super(Conv2D_BinaryLayer, self).build(input_shape)

    def call(self, inputs):
        self.b_kernel = binarization(self.kernel, self.H)

        outputs = tf.nn.conv2d(inputs, self.b_kernel, strides=self.strides, padding=self.padding,
                               data_format=self.data_format, dilations=self.dilation_rate)

        if self.use_bias:
            if self.data_format == 'channels_first':
                if len(self.kernel_size) == 1:
                    bias = tf.reshape(self.bias, (1, self.kernel_num, 1))
                    outputs += bias
                elif len(self.kernel_size) == 2:
                    outputs = tf.nn.bias_add(outputs, self.bias, data_format='NCHW')
                elif len(self.kernel_size) == 3:
                    outputs_shape = outputs.shape.as_list()
                    outputs_4d = tf.reshape(outputs, [outputs_shape[0], outputs_shape[1],
                                                     outputs_shape[2] * outputs_shape[3], outputs_shape[4]])
                    outputs_4d = tf.nn.bias_add(outputs_4d, self.bias, data_format='NCHW')
                    outputs = tf.reshape(outputs_4d, outputs_shape)
            else:
                outputs = tf.nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

class BatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 moving_mean_initializer='zeros', moving_variance_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                 gamma_constraint=None, renorm=False, renorm_clipping=None,
                 renorm_momentum=0.99, fused=None, **kwargs):
        super(BatchNormalization, self).__init__(
            axis=axis, momentum=momentum, epsilon=epsilon, center=center, scale=scale,
            beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
            renorm=renorm, renorm_clipping=renorm_clipping, renorm_momentum=renorm_momentum,
            fused=fused, **kwargs)

        self.axis = axis

    def build(self, input_shape):
        if self.fused is None:
            self.fused = True if self.axis in [1, -1] else False

        super(BatchNormalization, self).build(input_shape)

# Functional interface for the batch normalization layer.
def batch_normalization(inputs, axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True,
                        beta_initializer='zeros', gamma_initializer='ones',
                        moving_mean_initializer='zeros', moving_variance_initializer='ones',
                        beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                        gamma_constraint=None, training=False, trainable=True,
                        name=None, reuse=None, renorm=False, renorm_clipping=None,
                        renorm_momentum=0.99, fused=None):
    layer = BatchNormalization(
        axis=axis, momentum=momentum, epsilon=epsilon, center=center, scale=scale,
        beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
        renorm=renorm, renorm_clipping=renorm_clipping, renorm_momentum=renorm_momentum,
        fused=fused, trainable=trainable, name=name, dtype=inputs.dtype.base_dtype, _reuse=reuse,
        _scope=name)
    return layer(inputs, training=training)

