from glob import glob
import tensorflow as tf
import tensorflow.contrib as tf_contrib
from tensorflow.contrib.data import batch_and_drop_remainder
from data_processing.data_processing import ImageData

weight_init = tf_contrib.layers.variance_scaling_initializer(uniform=True)
weight_init_conv = tf.truncated_normal_initializer(stddev=0.001, seed=250)
weight_init_variable = tf.random_uniform_initializer(minval=0.0, maxval=0.001, seed = 250)

##################################################################################
# Data processing
##################################################################################
def processing_data(dir_B, dir_A, batch_size, h, w, ch, layers_num):
    trainA_dataset = glob(dir_A + '/*.*')
    trainB_dataset = glob(dir_B + '/*.*')

    dataset_num = max(len(trainA_dataset), len(trainB_dataset))

    Image_Data_Class = ImageData(h, w, ch)

    trainA = tf.data.Dataset.from_tensor_slices(trainA_dataset)
    trainB = tf.data.Dataset.from_tensor_slices(trainB_dataset)

    trainA = trainA.prefetch(batch_size).shuffle(dataset_num).map(Image_Data_Class.image_processing, num_parallel_calls=8).apply(batch_and_drop_remainder(batch_size)).repeat()
    trainB = trainB.prefetch(batch_size).shuffle(dataset_num).map(Image_Data_Class.image_processing, num_parallel_calls=8).apply( batch_and_drop_remainder(batch_size)).repeat()

    trainA_iterator = trainA.make_one_shot_iterator()
    trainB_iterator = trainB.make_one_shot_iterator()

    domain_A_info = trainA_iterator.get_next()
    domain_B_info = trainB_iterator.get_next()

    img_A, _ = domain_A_info
    img_B, _ = domain_B_info
    imgA_list = []
    imgB_list = []

    for layer in range(layers_num):
        if layer == layers_num-1:
            imgA_list.append(img_A)
            imgB_list.append(img_B)
        else:
            img_temp_A, img_temp_B = processing_By_config(img_A, img_B, h, w, layers_num, layer)
            imgA_list.append(img_temp_A)
            imgB_list.append(img_temp_B)
    return imgA_list, imgB_list

def processing_data_test(img, layers_num):
    img_list = []
    b, h, w, c = tf.unstack(tf.shape(img))
    for layer in range(layers_num):
        if layer == layers_num-1:
            img_list.append(img)
        else:
            factor = (layers_num - 1 - layer) * 2
            resize_shape = (h // factor, w // factor)
            img_temp = tf.image.resize_images(img, resize_shape)
            img_list.append(img_temp)
    return img_list

def processing_By_config(img_A, img_B, h, w, layers_num, layer):
    factor = (layers_num-1-layer)*2
    resize_shape = (tf.constant(h // factor), tf.constant(w // factor))
    img_A = tf.image.resize_images(img_A, resize_shape)
    img_B = tf.image.resize_images(img_B, resize_shape)
    return img_A, img_B


##################################################################################
# Layer
##################################################################################
def pad_reflect(x, padding=1):
    return tf.pad(
      x, [[0, 0], [padding, padding], [padding, padding], [0, 0]],
      mode='REFLECT')

def conv(x, channels, kernel=4, stride=2, pad=0, dilation_rate=(1, 1), pad_type='zero', use_norm=False,
         use_bias=True, scope='conv', use_relu=False, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):

        padding = 'valid'
        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
        if pad_type == 'same':
            padding = 'same'
        x = tf.layers.conv2d(inputs=x, filters=channels,dilation_rate=dilation_rate, padding=padding,
                             kernel_size=kernel, kernel_initializer=weight_init_conv,
                             strides=stride, use_bias=use_bias)
        if use_relu:
            x = lrelu(x)
        return x

def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

##################################################################################
# Normalization function
##################################################################################
def instance_norm(x, scope='instance_norm', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        return tf_contrib.layers.instance_norm(x, epsilon=1e-05, center=True, scale=True, scope=scope)

def layer_norm(x, scope='layer_norm') :
    return tf_contrib.layers.layer_norm(x, center=True, scale=True, scope=scope)

##################################################################################
# Activation function
##################################################################################
def lrelu(x, alpha=0.1):
    return tf.nn.leaky_relu(x, alpha)

def relu(x, scope=None):
    return tf.nn.relu(x, name=scope)

def tanh(x):
    return tf.tanh(x)

##################################################################################
# Residual-block
##################################################################################
def res_block_simple(x_input, use_bias=True, scope='simple_resblock', use_norm_layer=False, normType='Layer', reuse=False) :
    with tf.variable_scope(scope, reuse=reuse):
        channel = x_input.get_shape()[-1]
        x_init = x_input
        with tf.variable_scope('res1'):
            x = conv(x_init, channel, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            if use_norm_layer:
                if normType == 'Layer':
                    x = layer_norm(x)
                elif normType == 'IN':
                    x = instance_norm(x)

            x = lrelu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channel, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            if use_norm_layer:
                if normType == 'Layer':
                    x = layer_norm(x)
                elif normType == 'IN':
                    x = instance_norm(x)

        return x + x_init

def res_block_HDC(x, rate=1, use_relu=True, scope='res_block_HDC', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        x_init = x
        C = x_init.get_shape()[-1]

        x = conv(x, channels=C, kernel=1, stride=1, pad_type='reflect', dilation_rate=(1, 1), use_bias=False, scope='conv_0')
        x = lrelu(x)

        x = conv(x, channels=C, kernel=3, stride=1, pad_type='same', dilation_rate=(rate, rate), use_bias=False, scope='conv_1')
        x = lrelu(x)

        x = conv(x, channels=C, kernel=1, stride=1, pad_type='reflect', dilation_rate=(1, 1), use_bias=False, scope='conv_2')
        res_output = x + x_init

        if use_relu:
            res_output = lrelu(res_output)
        return  res_output

def res_block(x_input, channels=-1, reuse=False, use_norm_layer=True, normType='Layer', use_relu=True, bottle_neck=False, bottle_down_sample=True, scope="res_block"):
    with tf.variable_scope(scope, reuse=reuse):
        c_i = x_input.get_shape()[-1]

        if channels == -1:
            channels = c_i

        if not bottle_neck:
            first_output_channel = channels / 4
            second_output_channel = channels / 4
            third_output_channel = channels

            first_conv_output = conv(x_input, channels=first_output_channel, kernel=1, stride=1, pad_type='reflect',
                                     use_bias=False, scope='res_conv_0')
            if use_norm_layer:
                if normType == 'Layer':
                    first_conv_output = layer_norm(first_conv_output, scope='res_conv_norm_layer_0')
                elif normType == 'IN':
                    first_conv_output = instance_norm(first_conv_output, scope='res_conv_norm_layer_0')

            first_conv_output = lrelu(first_conv_output)

            second_conv_output = conv(first_conv_output, channels=second_output_channel, kernel=3, stride=1, pad_type='reflect', pad=1,
                                      use_bias=False, scope='res_conv_1')
            if use_norm_layer:
                if normType == 'Layer':
                    second_conv_output = layer_norm(second_conv_output, scope='res_conv_norm_layer_1')
                elif normType == 'IN':
                    second_conv_output = instance_norm(second_conv_output, scope='res_conv_norm_layer_1')


            second_conv_output = lrelu(second_conv_output)

            third_conv_output = conv(second_conv_output, channels=third_output_channel, kernel=1, stride=1, pad_type='reflect',
                                     use_bias=False, scope='res_conv_2')
            if use_norm_layer:
                if normType == 'Layer':
                    third_conv_output = layer_norm(third_conv_output, scope='res_conv_norm_layer_2')
                elif normType == 'IN':
                    third_conv_output = instance_norm(third_conv_output, scope='res_conv_norm_layer_2')

            if channels != c_i:
                x_input = conv(x_input, channels=third_output_channel, kernel=1, stride=1, pad_type='reflect', use_bias=False,scope='res_conv3')

            res_output = x_input + third_conv_output
            if use_relu:
                 res_output = lrelu(res_output)
        else:
            first_output_channel = channels / 2
            second_output_channel = channels / 2
            third_output_channel = channels

            first_conv_output = conv(x_input, channels=first_output_channel, kernel=1, stride=2, pad_type='reflect',
                                     use_bias=False, scope='res_conv_0')
            if use_norm_layer:
                if normType == 'Layer':
                    first_conv_output = layer_norm(first_conv_output, scope='res_conv_norm_layer_0')
                elif normType == 'IN':
                    first_conv_output = instance_norm(first_conv_output, scope='res_conv_norm_layer_0')

            first_conv_output = lrelu(first_conv_output)

            second_conv_output = conv(first_conv_output, channels=second_output_channel, kernel=3, stride=1,
                                      pad_type='reflect', pad=1, use_bias=False, scope='res_conv_1')
            if use_norm_layer:
                if normType == 'Layer':
                    second_conv_output = layer_norm(second_conv_output, scope='res_conv_norm_layer_1')
                elif normType == 'IN':
                    second_conv_output = instance_norm(second_conv_output, scope='res_conv_norm_layer_1')

            second_conv_output = lrelu(second_conv_output)

            third_conv_output = conv(second_conv_output, channels=third_output_channel, kernel=1, stride=1,
                                     pad_type='reflect', use_bias=False, scope='res_conv_2')
            if use_norm_layer:
                if normType == 'Layer':
                    third_conv_output = layer_norm(third_conv_output, scope='res_conv_norm_layer_2')
                elif normType == 'IN':
                    third_conv_output = instance_norm(third_conv_output, scope='res_conv_norm_layer_2')

            if bottle_down_sample:
                left_output = conv(x_input, channels=channels, kernel=1, stride=2, pad_type='zero', use_bias=False, scope='res_left_conv_0')
            else:
                left_output = conv(x_input, channels=channels, kernel=1, stride=1, pad_type='zero', use_bias=False, scope='res_left_conv_0')

            if use_norm_layer:
                if normType == 'Layer':
                    left_output = layer_norm(left_output, scope='res_conv_norm_layer_left')
                elif normType == 'IN':
                    left_output = instance_norm(left_output, scope='res_conv_norm_layer_left')


            res_output = left_output + third_conv_output
            if use_relu:
                res_output = lrelu(res_output)

        return res_output

##################################################################################
# Sampling
##################################################################################
def up_sample(x, scale_factor=2):
    _, h, w, _ = tf.unstack(tf.shape(x))
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

def down_sample(x, scale_factor=2):
    _, h, w, _ = tf.unstack(tf.shape(x))
    new_size = [h // scale_factor, w // scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

##################################################################################
# Loss
##################################################################################
def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss
def L2_loss(x):
    return tf.reduce_mean(tf.square(x))

def mse(x,y):
    '''Mean Squared Error'''
    return tf.reduce_mean(tf.square(x - y))

def sse(x,y):
    '''Sum of Squared Error'''
    return tf.reduce_sum(tf.square(x - y))

def gram_matrix(feature_maps):
    """Computes the Gram matrix for a set of feature maps."""
    batch_size, height, width, channels = tf.unstack(tf.shape(feature_maps))
    denominator = tf.to_float(height * width)
    feature_maps = tf.reshape(feature_maps, tf.stack([batch_size, height * width, channels]))
    matrix = tf.matmul(feature_maps, feature_maps, adjoint_a=True)
    return matrix / denominator

def gram_matrixs(vgg_maps):
    enc_style = [gram_matrix(s_map) for s_map in vgg_maps]
    return enc_style

def tf_cov(x,reduceMean = True):
    batch_size, height, width, channels = tf.unstack(tf.shape(x))
    x = tf.reshape(x, tf.stack([batch_size, height * width, channels]))
    mc = tf.reduce_mean(x, axis=1, keepdims=True)
    if reduceMean:
        fc = x - mc
    else:
        fc = x
    fcfc = tf.matmul(tf.transpose(fc,[0,2,1]),fc) / (tf.cast(tf.shape(fc)[1], tf.float32))
    return fcfc

def cov_matrixs(vgg_maps):
    enc_style = [tf_cov(s_map) for s_map in vgg_maps]
    return enc_style

##################################################################################
# Gradient
##################################################################################
def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads
