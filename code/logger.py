import tensorflow as tf 
import os

class Logger():
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args
        self.ckpt_dir = os.path.join(self.args.exp_dir, self.args.exp_name)
        self.scale_list = list(map(lambda x: int(x), self.args.scale.split('+')))
        if not args.is_test:
            self.summary_placeholders = {}
            self.summary_ops = {}
            self.train_writer = [tf.summary.FileWriter(os.path.join(self.ckpt_dir, "train_x%d" % scale)) for scale in self.scale_list]
            self.valid_writer = [tf.summary.FileWriter(os.path.join(self.ckpt_dir, "valid_x%d" % scale)) for scale in self.scale_list]
        
            print("Logger is constructed!")
            
    def write(self, summaries_dict, step, is_train, idx_scale, scope=""):
        # select training/validation
        if is_train:
            summary_writer = self.train_writer[idx_scale]
        else:
            summary_writer = self.valid_writer[idx_scale]

        if summaries_dict is not None:
            # make the list of summary (tf.summary.scalar + tf.scalar.image)
            summary_list = []
            for tag, value in summaries_dict.items():
                if tag not in self.summary_ops:
                    self.summary_placeholders[tag] = tf.placeholder('float32', value.shape, name=tag)
                    if len(value.shape) <= 1:
                        self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
                    else:
                        self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag])

                summary_list.append(self.sess.run(self.summary_ops[tag], {self.summary_placeholders[tag]: value}))
            # write the summaries
            for summary in summary_list:
                summary_writer.add_summary(summary, step)
            summary_writer.flush()