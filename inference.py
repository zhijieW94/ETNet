from network.Model import ModelNet
from network.ops import *
from utils import *

class Inference(object):
    def __init__(self, args):
        self.checkpoint_dir = args['checkpoint_dir']
        self.layers_num = args['layers_num']
        self.model = ModelNet(mode='test', args=args)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())
        for i in range(self.layers_num):
            layer_i = self.checkpoint_dir[i][0]
            dir_checkpoint = self.checkpoint_dir[i][1]
            if os.path.isdir(dir_checkpoint):
                ckpt = tf.train.get_checkpoint_state(dir_checkpoint)
                if ckpt and ckpt.model_checkpoint_path:
                    print("Restoring from checkpoint_%s..."%(str(i)))
                    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                    t_vars = tf.trainable_variables()
                    G_vars = [var for var in t_vars if ('layer_' + str(layer_i) + '/trainable' in var.name)]
                    self.saver = tf.train.Saver(var_list=G_vars)
                    self.saver.restore(self.sess, os.path.join(dir_checkpoint, ckpt_name))
                    print(" [*] Load checkpoint_%s SUCCESS"%(str(i)))
                else:
                    raise Exception("No checkpoint_%s found..."%(str(i)))

    def predict(self, img_content, img_style):

        fetches = {'img_fakes': self.model.img_fakes}

        s = time.time()
        results = self.sess.run(fetches, feed_dict={self.model.img_content: img_content, self.model.img_style: img_style})

        print("Stylized in:", time.time() - s)
        return results
