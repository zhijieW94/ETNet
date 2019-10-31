from network.ops import *
from keras.models import Model
from network.vgg_normalised import vgg_from_t7

class ModelNet(object):
    def __init__(self, mode='train', args=None):
        if mode == 'train':
            self.build_model_train(mode, args)
        else:
            self.build_model_test(mode, args)

    def build_model_test(self, mode, args):
        gpu_id = args['GPU_ID']
        with tf.device('/CPU:0'):
            print('build model...')
            self.img_contents = []
            self.img_styles = []
            self.img_fakes = []
            with tf.device('/GPU:%d' % gpu_id):
                print('tower:%d...' % gpu_id)
                with tf.name_scope('tower_%d' % gpu_id):
                    with tf.variable_scope('cpu_variables', reuse=gpu_id > 0):
                        img_fakes = self.build_model(mode, args, gpu_id)
                        self.img_fakes.append(img_fakes)
            print('build model on gpu tower done.')
            print('build model done...')

    def build_model_train(self, mode, args):
        gpu_id_list = args['GPU_ID']
        batch_size = args['batch_size']
        nums_gpu = len(gpu_id_list)
        self.batch_size = batch_size//nums_gpu
        with tf.device('/CPU:0'):
            self.lr = tf.placeholder(tf.float32, name='learning_rate')
            opt = tf.train.AdamOptimizer(learning_rate=self.lr)
            print('build model...')
            print('build model on gpu tower...')
            tower_grads = []
            self.img_contents = []
            self.img_styles = []
            self.img_fakes = []
            for gpu_id in range(nums_gpu):
                with tf.device('/GPU:%d' % gpu_id):
                    print('tower:%d...' % gpu_id)
                    with tf.name_scope('tower_%d' % gpu_id):
                        with tf.variable_scope('cpu_variables', reuse=gpu_id > 0):
                            T_vars, grads, loss_list, img_contents, img_styles, img_fakes = self.build_model(mode, args, gpu_id)
                            tower_grads.append(zip(grads, T_vars))
                            self.loss_list = loss_list
                            self.img_contents.append(img_contents)
                            self.img_styles.append(img_styles)
                            self.img_fakes.append(img_fakes)
            print('build model on gpu tower done.')
            print('reduce model on cpu...')
            grads = average_gradients(tower_grads)
            tf.summary.scalar('learning_rate', self.lr)
            tf.contrib.training.add_gradients_summaries(grads)
            optim = opt.apply_gradients(grads)
            self.optim = optim
            self.summary = tf.summary.merge_all()
            print('build model done...')

    def build_model(self, mode, args, gpu_id=0):
        vgg_weights = args['vgg_weights']
        layers_num = args['layers_num']

        '''data processing '''
        img_contents, img_styles = self.data_processing(mode, args)

        '''build model '''
        self.Build_vgg_model(vgg_weights, reuse=tf.AUTO_REUSE)

        img_fake = tf.random.truncated_normal(tf.shape(img_contents[0]), mean=0.0, stddev=0.1, dtype=tf.float32)
        # img_fake = tf.zeros(tf.shape(img_contents[0]))
        img_fakes = []
        for i_layer in range(layers_num):
            if i_layer == 0:
                delta_content, _ = self.get_delta_content_list(img_fake, img_contents[0], img_styles[0])
                delta_content_list = [delta_content]
                delta_style_list, vgg_fake_list = self.get_delta_style_list(img_fake, img_styles[0])
            elif i_layer == 1:
                delta_content_0, delta_style_0 = self.get_delta_content_list(img_fake, img_contents[0], img_styles[0])
                img_fake = up_sample(img_fake)
                delta_content_1, _ = self.get_delta_content_list(img_fake, img_contents[1], img_styles[1])
                delta_content_list = [delta_content_0, delta_content_1]

                delta_style_list, vgg_fake_list = self.get_delta_style_list(img_fake, img_styles[1])
                delta_style_list.append(delta_style_0)
            elif i_layer == 2:
                img_fake_0 = down_sample(img_fake)
                delta_content_0, delta_style_0 = self.get_delta_content_list(img_fake_0, img_contents[0], img_styles[0])
                delta_content_1, delta_style_1 = self.get_delta_content_list(img_fake, img_contents[1], img_styles[1])

                img_fake = up_sample(img_fake)
                delta_content_2, _ = self.get_delta_content_list(img_fake, img_contents[2], img_styles[2])
                delta_style_list, vgg_fake_list = self.get_delta_style_list(img_fake, img_styles[2])

                delta_content_list = [delta_content_0, delta_content_1, delta_content_2]
                delta_style_list.append(delta_style_1)
                delta_style_list.append(delta_style_0)

            scope_name = 'layer_' + str(i_layer) + '/trainable'
            with tf.variable_scope(scope_name):
                if i_layer == 0:
                    error_content_relu4_1 = delta_content_list[0]
                elif i_layer == 1:
                    error_content_relu5_1 = delta_content_list[0]
                    error_style_relu5_1 = delta_style_list[4]
                    error_Fusion_relu5_1 = self.Fusion_simple(error_content_relu5_1, error_style_relu5_1, channel=512, scope='fusion_error_relu5_1', reuse=tf.AUTO_REUSE)
                    error_content_relu4_1 = self.Decoder(error_Fusion_relu5_1, relu_target='relu5_1', scope='dec_relu5_1', reuse=tf.AUTO_REUSE)
                    error_content_relu4_1 = error_content_relu4_1 + delta_content_list[1]

                elif i_layer == 2:
                    error_content_relu6_1 = delta_content_list[0]
                    error_style_relu6_1 = delta_style_list[5]
                    error_Fusion_relu6_1 = self.Fusion_simple(error_content_relu6_1, error_style_relu6_1, channel=512, scope='fusion_error_relu6_1', reuse=tf.AUTO_REUSE)
                    error_content_relu5_1 = self.Decoder(error_Fusion_relu6_1, relu_target='relu6_1', scope='dec_relu6_1', reuse=tf.AUTO_REUSE)

                    error_content_relu5_1 = error_content_relu5_1 + delta_content_list[1]
                    error_style_relu5_1 = delta_style_list[4]
                    error_Fusion_relu5_1 = self.Fusion_simple(error_content_relu5_1, error_style_relu5_1, channel=512, scope='fusion_error_relu5_1', reuse=tf.AUTO_REUSE)
                    error_content_relu4_1 = self.Decoder(error_Fusion_relu5_1, relu_target='relu5_1', scope='dec_relu5_1', reuse=tf.AUTO_REUSE)
                    error_content_relu4_1 = error_content_relu4_1 + delta_content_list[2]

                error_style_relu4_1 = delta_style_list[3]
                error_Fusion_relu4_1 = self.Fusion_simple(error_content_relu4_1, error_style_relu4_1, channel=512, scope='fusion_error_relu4_1', reuse=tf.AUTO_REUSE)
                error_content_relu3_1 = self.Decoder(error_Fusion_relu4_1, relu_target='relu4_1', scope='dec_relu4_1', reuse=tf.AUTO_REUSE)

                error_style_relu3_1 = delta_style_list[2]
                error_Fusion_relu3_1 = self.Fusion_simple(error_content_relu3_1, error_style_relu3_1, channel=256, scope='fusion_error_relu3_1', reuse=tf.AUTO_REUSE)
                error_content_relu2_1 = self.Decoder(error_Fusion_relu3_1, relu_target='relu3_1', scope='dec_relu3_1', reuse=tf.AUTO_REUSE)

                error_style_relu2_1 = delta_style_list[1]
                error_Fusion_relu2_1 = self.Fusion_simple(error_content_relu2_1, error_style_relu2_1, channel=128, scope='fusion_error_relu2_1', reuse=tf.AUTO_REUSE)
                error_content_relu1_1 = self.Decoder(error_Fusion_relu2_1, relu_target='relu2_1', scope='dec_relu2_1', reuse=tf.AUTO_REUSE)

                error_style_relu1_1 = delta_style_list[0]
                error_Fusion_relu1_1 = self.Fusion_simple(error_content_relu1_1, error_style_relu1_1, channel=64,
                                                          scope='fusion_error_relu1_1', reuse=tf.AUTO_REUSE)

                fake_content_relu4_1 = vgg_fake_list[3]
                fake_diffusion_relu4_1 = self.Diffusion(error_Fusion_relu4_1, fake_content_relu4_1, scope='diffusion_relu4_1', reuse=tf.AUTO_REUSE)

                fake_content_relu3_1 = self.Decoder(fake_diffusion_relu4_1, relu_target='relu4_1', scope='dec_relu4_1', reuse=tf.AUTO_REUSE)

                fake_content_relu3_1 = lrelu(fake_content_relu3_1 + vgg_fake_list[2] + error_Fusion_relu3_1)
                fake_content_relu2_1 = self.Decoder(fake_content_relu3_1, relu_target='relu3_1', scope='dec_relu3_1', reuse=tf.AUTO_REUSE)

                fake_content_relu2_1 = lrelu(fake_content_relu2_1 + vgg_fake_list[1] + error_Fusion_relu2_1)
                fake_content_relu1_1 = self.Decoder(fake_content_relu2_1, relu_target='relu2_1', scope='dec_relu2_1', reuse=tf.AUTO_REUSE)

                fake_content_relu1_1 = lrelu(fake_content_relu1_1 + vgg_fake_list[0] + error_Fusion_relu1_1)
                fake_content = self.Decoder(fake_content_relu1_1, relu_target='relu1_1', scope='dec_relu1_1', reuse=tf.AUTO_REUSE)

                img_fake = tanh(fake_content + img_fake)
                img_fakes.append(img_fake)

        if mode == 'train':
            incremental = args['Incremental']
            weight_content = args['weight']['content']
            weight_style = args['weight']['style']

            '''loss ops'''
            loss_content, loss_style = self.get_pLoss_multiscale(img_contents, img_styles, img_fakes[-1], layers_num)
            loss = weight_content * loss_content + weight_style * loss_style

            '''Training ops'''
            vars = tf.trainable_variables()
            t_vars = [var for var in vars if 'layer_' + str(incremental) + '/trainable' in var.name]

            grads = tf.gradients(loss, t_vars)
            grads, _ = tf.clip_by_global_norm(grads, 1)

            '''Summary ops'''
            if gpu_id == 0:
                tf.summary.scalar('loss_content', loss_content)
                tf.summary.scalar('loss_style', loss_style)
            loss_list = [loss, loss_content, loss_style]
            return t_vars, grads, loss_list, img_contents, img_styles, img_fakes
        else:
            return img_fakes


    def Build_vgg_model(self, vgg_weights, reuse=False):
        with tf.variable_scope('vgg_model', reuse=reuse):
            self.vgg_model = vgg_from_t7(vgg_weights, target_layer='relu4_1')

            relu_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']
            model_layers = [self.vgg_model.get_layer(l).output for l in relu_layers]
            self.vgg_layers_model = Model(inputs=self.vgg_model.input, outputs=model_layers)

            content_layer = self.vgg_model.get_layer('relu4_1').output
            self.vgg_lastLayer_model = Model(inputs=self.vgg_model.input, outputs=content_layer)


    def get_pLoss_multiscale(self, content_imgs, style_imgs, fake_img, layers_num, reuse=False, scope='loss_p'):
        with tf.variable_scope(scope, reuse=reuse):
            if layers_num == 1:
                fake_fmaps = self.vgg_layers_model(fake_img)
                style_fmaps = self.vgg_layers_model(style_imgs[0])
                content_fmap = self.vgg_lastLayer_model(content_imgs[0])
                content_loss = mse(fake_fmaps[-1], content_fmap)
                style_loss, _ = self.get_style_loss(fake_fmaps, style_fmaps)

            elif layers_num == 2:
                fake_0 = down_sample(fake_img)
                fake_1 = fake_img

                fake_fmaps = self.vgg_layers_model(fake_0)
                style_fmaps = self.vgg_layers_model(style_imgs[0])
                content_fmap = self.vgg_lastLayer_model(content_imgs[0])
                content_loss_0 = mse(fake_fmaps[-1], content_fmap)
                style_loss_0, _ = self.get_style_loss(fake_fmaps, style_fmaps)

                fake_fmaps = self.vgg_layers_model(fake_1)
                style_fmaps = self.vgg_layers_model(style_imgs[1])
                content_fmap = self.vgg_lastLayer_model(content_imgs[1])
                content_loss_1 = mse(fake_fmaps[-1], content_fmap)
                style_loss_1, style_loss_last = self.get_style_loss(fake_fmaps, style_fmaps)

                content_loss = content_loss_0 + content_loss_1
                style_loss = style_loss_0 + style_loss_1 + 6 * style_loss_last

            elif layers_num == 3:
                fake_2 = fake_img
                fake_1 = down_sample(fake_img)
                fake_0 = down_sample(fake_img, scale_factor=4)

                fake_fmaps = self.vgg_layers_model(fake_0)
                style_fmaps = self.vgg_layers_model(style_imgs[0])
                content_fmap = self.vgg_lastLayer_model(content_imgs[0])
                content_loss_0 = mse(fake_fmaps[-1], content_fmap)
                style_loss_0, _ = self.get_style_loss(fake_fmaps, style_fmaps)

                fake_fmaps = self.vgg_layers_model(fake_1)
                style_fmaps = self.vgg_layers_model(style_imgs[1])
                content_fmap = self.vgg_lastLayer_model(content_imgs[1])
                content_loss_1 = mse(fake_fmaps[-1], content_fmap)
                style_loss_1, _ = self.get_style_loss(fake_fmaps, style_fmaps)

                fake_fmaps = self.vgg_layers_model(fake_2)
                style_fmaps = self.vgg_layers_model(style_imgs[2])
                content_fmap = self.vgg_lastLayer_model(content_imgs[2])
                content_loss_2 = mse(fake_fmaps[-1], content_fmap)
                style_loss_2, style_loss_last = self.get_style_loss(fake_fmaps, style_fmaps)

                content_loss = content_loss_0 + content_loss_1 + content_loss_2
                style_loss = style_loss_0 + style_loss_1 + style_loss_2 + 10 * style_loss_last

            return content_loss, style_loss


#code = content*weight*style
    def Fusion_simple(self, content, style, channel=512, scope='fusion', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            _, w, h, _ = tf.unstack(tf.shape(content))
            weight = tf.get_variable(name='weight', shape=[channel, channel], initializer=weight_init_variable)
            weight = tf.tile(tf.expand_dims(weight, axis=0), [self.batch_size, 1, 1])
            content = res_block(content, channels=channel, bottle_neck=False, use_norm_layer=False, use_relu=False, scope='res_block')
            content = tf.reshape(content, shape=[self.batch_size, w * h, channel])
            code_fusion = tf.matmul(weight, style)
            code_fusion = tf.matmul(content, code_fusion)
            code_fusion = tf.reshape(code_fusion, shape=[self.batch_size, w, h, channel])
            code_fusion = res_block(code_fusion, channels=channel, bottle_neck=False, use_norm_layer=False, use_relu=True, scope='res_block')
            # code_fusion = res_block_simple(code_fusion, scope='res_block')
            return code_fusion

    def Decoder(self, x, relu_target, scope='decoder', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            if relu_target == 'relu6_1':
                with tf.variable_scope('relu6_1', reuse=reuse):
                    x = res_block(x, channels=128, bottle_neck=False, use_norm_layer=False, use_relu=True, scope='res_block_0')
                    x = res_block_simple(x, use_bias=True, scope='res_block_1', use_norm_layer=False)
                    x = up_sample(x, scale_factor=2)
                    x = res_block_simple(x, use_bias=True, scope='res_block_2', use_norm_layer=False)
                    x = lrelu(x)
                    x = res_block(x, channels=512, bottle_neck=False, use_norm_layer=False, use_relu=True, scope='res_block_3')

            if relu_target == 'relu5_1':
                with tf.variable_scope('relu5_1', reuse=reuse):
                    x = res_block(x, channels=128, bottle_neck=False, use_norm_layer=False, use_relu=True, scope='res_block_0')
                    x = res_block_simple(x, use_bias=True, scope='res_block_1', use_norm_layer=False)
                    x = up_sample(x, scale_factor=2)
                    x = res_block_simple(x, use_bias=True, scope='res_block_2', use_norm_layer=False)
                    x = lrelu(x)
                    x = res_block(x, channels=512, bottle_neck=False, use_norm_layer=False, use_relu=True, scope='res_block_3')

            if relu_target == 'relu4_1':
                with tf.variable_scope('relu4_1', reuse=reuse):
                    x = res_block(x, channels=256, bottle_neck=False, use_norm_layer=False, use_relu=True, scope='res_block_0')
                    x = res_block_simple(x, use_bias=True, scope='res_block_1', use_norm_layer=False)
                    x = up_sample(x, scale_factor=2)
                    x = res_block_simple(x, use_bias=True, scope='res_block_2', use_norm_layer=False)

            if relu_target == 'relu3_1':
                with tf.variable_scope('relu3_1', reuse=reuse):
                    x = conv(x, 128, kernel=3, stride=1, pad=1, pad_type='reflect', use_relu=True, scope='conv')
                    x = res_block_simple(x, use_bias=True, scope='res_block_1', use_norm_layer=False)
                    x = up_sample(x, scale_factor=2)
                    x = res_block_simple(x, use_bias=True, scope='res_block_2', use_norm_layer=False)

            if relu_target == 'relu2_1':
                with tf.variable_scope('relu2_1', reuse=reuse):
                    x = conv(x, 128, kernel=3, stride=1, pad=1, pad_type='reflect', use_relu=True, scope='conv_0')
                    x = res_block_simple(x, use_bias=True, scope='res_block_1', use_norm_layer=False)
                    x = up_sample(x, scale_factor=2)
                    x = res_block_simple(x, use_bias=True, scope='res_block_2', use_norm_layer=False)
                    x = lrelu(x)
                    x = conv(x, 64, kernel=3, stride=1, pad=1, pad_type='reflect', use_relu=True, scope='conv')

            if relu_target == 'relu1_1':
                x = res_block_simple(x, use_bias=True, scope='res_block_1', use_norm_layer=False)
                x = lrelu(x)
                x = res_block_simple(x, use_bias=True, scope='res_block_2', use_norm_layer=False)
                x = lrelu(x)
                x = conv(x, 64, kernel=3, stride=1, pad=1, pad_type='reflect', use_relu=True, scope='conv')
                x = conv(x, 3, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv_last')
            return x

    def Diffusion(self, error, target, scope='diffusion', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            _, w, h, C = error.get_shape().as_list()
            error = res_block(error, channels=C, bottle_neck=False, use_norm_layer=False, use_relu=False, scope='res_block_error')
            target = res_block(target, channels=C, bottle_neck=False, use_norm_layer=False, use_relu=False, scope='res_block_target')
            target_diffused = self.Attn_block(error, target, scope='error2target')
            error_diffused = self.Attn_block(target, error, scope='target2error')
            code_diffused = lrelu(target_diffused + error_diffused)
            # code_diffused = res_block_simple(code_diffused,  scope='res_block')
            code_diffused = res_block(code_diffused, channels=C, bottle_neck=False, use_norm_layer=False, use_relu=True, scope='res_block')
            return code_diffused

    def Attn_block(self, source, target, scope='diffusion_block', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            _, w, h, C = tf.unstack(tf.shape(source))
            C = source.shape[3]
            f = conv(source, C//2, kernel=1, stride=1, pad=0, pad_type='reflect', scope='f_conv') # [bs, h, w, c']
            g = conv(target, C//2, kernel=1, stride=1, pad=0, pad_type='reflect', scope='g_conv') # [bs, h, w, c']
            h = conv(target, C//2, kernel=1, stride=1, pad=0, pad_type='reflect', scope='h_conv') # [bs, h, w, c']
            s = tf.matmul(hw_flatten(f), hw_flatten(g),transpose_b=True) # [bs, N, N]
            beta = tf.nn.softmax(s)  # self_attention map
            o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
            o = tf.reshape(o, shape=tf.shape(h))  # [bs, h, w, c]
            o = conv(o, C, kernel=1, stride=1, pad=0, pad_type='reflect', scope='out_conv')
            o = o + target
            return o

    def get_delta_content_list(self, img_fake, img_content, img_style):
        vgg_fake = self.vgg_lastLayer_model(img_fake)
        vgg_content = self.vgg_lastLayer_model(img_content)
        vgg_style = self.vgg_lastLayer_model(img_style)
        gram_style = gram_matrix(vgg_style)
        gram_fake = gram_matrix(vgg_fake)
        delta_style = tf.sqrt(gram_fake) - tf.sqrt(gram_style)
        return vgg_fake - vgg_content, delta_style

    def get_delta_style_list(self, img_fake, img_style):
        vgg_fake_list = self.vgg_layers_model(img_fake)
        vgg_style_list = self.vgg_layers_model(img_style)

        gram_style_list = gram_matrixs(vgg_style_list)
        gram_fake_list = gram_matrixs(vgg_fake_list)

        delta_style_list = [tf.sqrt(gram_fake) - tf.sqrt(gram_style) for (gram_style, gram_fake) in zip(gram_style_list, gram_fake_list)]
        return delta_style_list, vgg_fake_list

    def get_style_loss(self, fake_maps, style_maps):
        gram_losses = []
        for s_map, d_map in zip(style_maps, fake_maps):
            s_gram = gram_matrix(s_map)
            d_gram = gram_matrix(d_map)
            gram_loss = mse(d_gram, s_gram)
            gram_losses.append(gram_loss)
        style_loss = tf.reduce_sum(gram_losses) / self.batch_size
        return style_loss, gram_losses[0]


    def data_processing(self, mode, args):
        dir_style = args['data']['dir_style']
        dir_content = args['data']['dir_content']
        layers_num = args['layers_num']
        if mode == 'train':
            img_size = args['img_size']
            img_contents, img_styles = processing_data(dir_style, dir_content, self.batch_size, img_size[0],
                                                       img_size[1], img_size[2], layers_num)
        else:
            self.batch_size = 1
            self.img_content = tf.placeholder(tf.float32, [1, None, None, 3], name='test_image')
            self.img_style = tf.placeholder(tf.float32, [1, None, None, 3], name='test_style')
            img_contents = processing_data_test(self.img_content, layers_num)
            img_styles = processing_data_test(self.img_style, layers_num)

        return img_contents, img_styles