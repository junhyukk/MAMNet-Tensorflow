from options import args
from data import DataGenerator
from logger import Logger
from trainer import Trainer
import tensorflow as tf
import os 
from utils import create_dirs
from importlib import import_module
import win_unicode_console

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
win_unicode_console.enable()

def main():
    # create the experiments directory
    ckpt_dir = os.path.join(args.exp_dir, args.exp_name)
    results_dir = os.path.join(ckpt_dir, 'results')
    create_dirs([ckpt_dir, results_dir])

    # create tensorflow session
    sess = tf.Session()
    print("\nSession is created!")

    # create instances of the model, data generator, logger, and trainer
    module = import_module("model." + args.model_name)
    model = module.create_model(args)

    if args.is_train or args.is_test:
        data = DataGenerator(args) 
        logger = Logger(sess, args) 
        trainer = Trainer(sess, model, data, logger, args) 

    if args.is_train:
        trainer.train()
    if args.is_test: 
        trainer.test() 

if __name__ == '__main__':
    main()