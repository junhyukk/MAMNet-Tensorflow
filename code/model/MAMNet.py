from model.base_model import BaseModel
import tensorflow as tf

def create_model(args, parent=False):
    return MAMNet(args)

class MAMNet(BaseModel):
	def __init__(self, args):
		super(MAMNet, self).__init__(args)
		self.build_model()
		self.init_saver()
		num_params = self.count_num_trainable_params()
		print("num_params: %d" % num_params)
		print("MAMNet and Saver are initialized!")

	def build_model(self):
		# Initialization
		num_feats = self.args.num_feats
		num_res = self.args.num_res
		mean_shift = self.mean_shift
		conv_xavier = self.conv_xavier
		res_module = self.res_module
		scale_specific_upsampler = self.scale_specific_upsampler

		# Placeholder for input, target, and flag_scale
		if self.args.is_test:
			self.input = tf.placeholder(tf.float32, [1, None, None, 3])
		else:
			self.input = tf.placeholder(tf.float32, [self.args.num_batch, self.args.patch_size, self.args.patch_size, 3])
			self.target = tf.placeholder(tf.float32, [self.args.num_batch, self.args.patch_size*int(self.args.scale), self.args.patch_size*int(self.args.scale), 3])

		# Pre-processing
		in_img = mean_shift(self.input)

    	# First convolution layer 
		x = conv_xavier(in_img, num_feats, [3,3])
		x = res_module(x, num_feats, num_res)
		x = scale_specific_upsampler(x, int(self.args.scale))

		# Post-processing
		self.output = mean_shift(x, is_add=True) 

		if self.args.is_train:
			# Loss & Training options
			with tf.name_scope("loss"):
				self.loss = tf.reduce_mean(tf.losses.absolute_difference(self.target, self.output))
				self.learning_rate = tf.train.exponential_decay(self.args.init_lr, self.global_step, self.args.decay_step, self.args.decay_ratio, staircase=True)
				self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
