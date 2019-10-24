from network.Model import ModelNet as model_net
from utils import *
from network.ops import *
from data_processing.data_processing import save_images
import numpy as np
from scipy import misc

class Train(object):
    def __init__(self, args):
        self.gpu_id = args['GPU_ID']
        self.epoch = args['epoch']
        self.iteration = args['iteration']
        self.batch_size = args['batch_size']

        self.init_lr   = args['lr']
        self.lr_decay  = args['lr_decay']

        self.print_freq = args['freq_print']
        self.save_freq  = args['freq_save']
        self.log_freq  = args['freq_log']

        self.layers_num = args['layers_num']
        self.Incremental = args['Incremental']
        self.img_size = args['img_size']

        self.checkpoint_dir_load = args['dir_checkpoint']

        '''build model'''
        self.model = model_net(mode='train', args=args)

        ''' build folders for saving results'''
        self.log_dir, self.config_dir, self.sample_dir, self.checkpoint_dir = mkdir_output_train(args)

        ''' load model'''
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True))
        self.writer = tf.summary.FileWriter(self.log_dir + '/', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        could_load, checkpoint_counter = self.loadCheckpoint()

        if could_load:
            self.start_epoch = (int)(checkpoint_counter / self.iteration)
            self.start_batch_id = checkpoint_counter - self.start_epoch * self.iteration
            self.counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            self.start_epoch = 0
            self.start_batch_id = 0
            self.counter = 1
            print(" [!] Load failed...")

        layer_i = self.Incremental
        vars = tf.trainable_variables()
        t_vars = [var for var in vars if ('layer_' + str(layer_i) + '/trainable' in var.name)]
        self.saver = tf.train.Saver(var_list=t_vars)

        self.fetches = {
            'img_content': self.model.img_contents,
            'img_style': self.model.img_styles,
            'img_fakes': self.model.img_fakes,
            'train': self.model.optim,
            'loss_list': self.model.loss_list,
            'summary': self.model.summary
        }

    def train(self):
        for epoch in range(self.start_epoch, self.epoch):
            for idx in range(self.start_batch_id, self.iteration):
                lr = self.init_lr / (1 + self.counter * self.lr_decay)
                start_time = time.time()
                results = self.sess.run(self.fetches, feed_dict={self.model.lr: lr})
                self.counter += 1

                #print losses
                print("GPU_id:[%s] Epoch: [%2d] [%6d/%6d] time: %4.4f loss: %.8f c_loss: %.8f s_loss: %.8f"\
                    % (''.join(str(x) for x in self.gpu_id), epoch, idx, self.iteration, time.time() - start_time, results['loss_list'][0], results['loss_list'][1], results['loss_list'][2]))

                #save summary
                if np.mod(idx, self.log_freq) == 0:
                    self.writer.add_summary(results['summary'], self.counter)

                # save images
                if np.mod(idx, self.print_freq) == 0:
                    list_img_temp = []
                    for id in range(len(self.gpu_id)):
                        img_contents = results['img_content'][id]
                        img_fakes = results['img_fakes'][id]
                        img_styles = results['img_style'][id]

                        for i in range(self.batch_size//len(self.gpu_id)):
                            list_img_temp.append(img_contents[-1][i,:,:,:])
                            for j in range(len(img_fakes)):
                                img = img_fakes[j][i]
                                dim1, dim2 = img.shape[0], img.shape[1]
                                if dim1 != self.img_size[0] or dim2!=self.img_size[1]:
                                    img = (img + 1.) / 2
                                    img = np.uint8(np.clip(img, 0, 1) * 255)
                                    img = np.clip(img, 0, 255).astype(np.uint8)
                                    img = misc.imresize(img, [self.img_size[0], self.img_size[1]])
                                    img = img / 127.5 - 1
                                list_img_temp.append(img)

                            list_img_temp.append(img_styles[-1][i, :, :, :])


                    array_img_out = np.array(list_img_temp, dtype=np.float32)
                    num = int(len(list_img_temp) / self.batch_size)
                    save_images(array_img_out, [self.batch_size * num, num], '{}/{:02d}_{:06d}.jpg'.format(self.sample_dir, epoch, idx))

                #save model
                if np.mod(idx + 1, self.save_freq) == 0:
                    self.saveCheckpoint(self.counter)

            self.start_batch_id = 0

            # save model for final step
            self.saveCheckpoint(self.counter)
        print("finish...!")

    def loadCheckpoint(self):
        import re
        print(" [*] Reading checkpoints...")
        try:
            counter = 0
            for i in range(len(self.checkpoint_dir_load)):
                layer_i = self.checkpoint_dir_load[i][0]
                dir_checkpoint = self.checkpoint_dir_load[i][1]

                ckpt = tf.train.get_checkpoint_state(dir_checkpoint)
                if ckpt and ckpt.model_checkpoint_path:
                    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                    vars = tf.trainable_variables()
                    t_vars = [var for var in vars if ('layer_' + str(layer_i) + '/trainable' in var.name)]
                    self.saver = tf.train.Saver(var_list=t_vars)
                    self.saver.restore(self.sess, os.path.join(dir_checkpoint, ckpt_name))
                    if layer_i == self.Incremental:
                        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
                    print(" [*] Success to read {}".format(ckpt_name))
            print(" [*] Success to load checkpoint!!!")
            return True, counter
        except:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def saveCheckpoint(self, step):
        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'Res_Transfer_Net.model'), global_step=step)

