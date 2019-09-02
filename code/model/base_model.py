import tensorflow as tf
import os, math

class BaseModel:
    def __init__(self, args):
        self.args = args
        self.init_global_step()
        self.ckpt_dir = os.path.join(self.args.exp_dir, self.args.exp_name)

    def conv(self, x, num_feats, kernel_size=[3,3], strides=(1,1), padding='same', activation=None, kernel_initializer=None):
        if self.args.is_init_res:
            scale_list = list(map(lambda x: int(x), self.args.scale.split('+')))
            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(factor=0.0001/self.args.num_res, mode="FAN_IN", uniform=False)
        elif self.args.is_init_he:
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
        return tf.layers.conv2d(x, num_feats, kernel_size, strides=strides, padding=padding, activation=activation, kernel_initializer=kernel_initializer)

    def conv_xavier(self, x, num_feats, kernel_size, padding='same', activation=None):
        return tf.layers.conv2d(x, num_feats, kernel_size, padding=padding, activation=activation)

    def conv_he(self, x, num_feats, kernel_size, padding='same', activation=None):
        return tf.layers.conv2d(x, num_feats, kernel_size, padding=padding, activation=activation, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    def mam(self, x, num_feats, ratio=16):
        modulation_map_CSI = 0.0
        modulation_map_ICD = 0.0
        modulation_map_CSD = 0.0

        if self.args.is_CSI or self.args.is_ICD:
            _, tmp_var = tf.nn.moments(x, axes=[1,2], keep_dims=True)
            if self.args.is_std_norm:
                mean_var, var_var = tf.nn.moments(tmp_var, axes=-1, keep_dims=True)
                tmp_var = (tmp_var - mean_var) / tf.sqrt(var_var + 1e-5)

        if self.args.is_CSI:
            with tf.variable_scope('CSI'):
                modulation_map_CSI = tmp_var

        if self.args.is_ICD:
            with tf.variable_scope('ICD'):
                tmp = tf.layers.dense(tmp_var, num_feats//ratio, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
                modulation_map_ICD = tf.layers.dense(tmp, num_feats)

        if self.args.is_CSD:
            with tf.variable_scope('CSD'):
                W = tf.get_variable("W", shape=[3,3,num_feats,1], initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b", shape=[num_feats], initializer=tf.zeros_initializer)       
                modulation_map_CSD = tf.nn.depthwise_conv2d(x, filter=W, strides=[1,1,1,1], padding='SAME') + b

        modulation_map = tf.sigmoid(modulation_map_CSI+modulation_map_ICD+modulation_map_CSD)
        return modulation_map * x

    def res_block(self, x, num_feats, kernel_size, name='RB'):
        with tf.variable_scope(name):
            tmp = self.conv(x, num_feats, kernel_size, activation=tf.nn.relu)
            tmp = self.conv(tmp, num_feats, kernel_size)
            if self.args.is_MAM:
                tmp = self.mam(tmp, num_feats)
        return x + tmp

    def res_module(self, x, num_feats, num_res, name='RM'):
        with tf.variable_scope(name):
            before_res = x
            for _ in range(num_res):
                x = self.res_block(x, num_feats, [3,3], name='RB'+str(_))
            x = self.conv(x, num_feats, [3,3])
        return before_res + x

    def mean_shift(self, x, is_add=False):
        mean_vec = [0.4488, 0.4371, 0.4040]
        mean_vec = [x * 255 for x in mean_vec] 
        if is_add:
            x = x + mean_vec
        else:
            x = x - mean_vec
        return x

    def scale_specific_upsampler(self, x, scale):
        x = self.upsampler(x, scale)
        x = self.conv_xavier(x, self.args.num_channels, [3,3])
        return x

    # Method to upscale an image using conv2d transpose.
    def upsampler(self, x, scale):
        with tf.variable_scope('upsampler'+str(scale)):
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log(scale, 2))):
                    x = self.conv_xavier(x, self.args.num_feats*4, [3,3])
                    x = tf.depth_to_space(x, 2)
                return x
            elif scale == 3:
                x = self.conv_xavier(x, self.args.num_feats*9, [3,3])
                x = tf.depth_to_space(x, 3)
                return x
            else:
                raise NotImplementedError              

    # save function thet save the checkpoint in the path defined in argsfile
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.ckpt_dir+"/model.ckpt", self.global_step)
        print("Model saved")

    # load lateset checkpoint from the experiment path defined in args_file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.ckpt_dir)
        if self.args.ckpt_name:
            self.ckpt = self.ckpt_dir+"/"+self.args.ckpt_name
        else:
            self.ckpt = latest_checkpoint
        print("Loading model checkpoint {} ...\n".format(self.ckpt))
        self.saver.restore(sess, self.ckpt)
        print("Model loaded")

    # just inialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.args.max_to_keep)

    def count_num_trainable_params(self):
        tot_nb_params = 0
        for trainable_variable in tf.trainable_variables():
            shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
            current_nb_params = self.get_num_params_shape(shape)
            tot_nb_params = tot_nb_params + current_nb_params
        return tot_nb_params

    def get_num_params_shape(self, shape):
        nb_params = 1
        for dim in shape:
            nb_params = nb_params*int(dim)
        return nb_params 