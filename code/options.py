import argparse

# Parameter setting 
parser = argparse.ArgumentParser()

# Device option
parser.add_argument("--device", default='0')

# Directories
parser.add_argument("--data_dir", default="D:/Dataset/DIV2K")
parser.add_argument("--exp_dir", default="D:/KJH/tensorflow/experiments")
parser.add_argument("--exp_name", default="MAMNet")
parser.add_argument("--model_name", default="MAMNet")

# Model specifications 
parser.add_argument("--patch_size", default=48, type=int)
parser.add_argument("--num_channels", default=3, type=int)
parser.add_argument("--num_feats", default=64, type=int)
parser.add_argument("--num_res", default=16, type=int)
parser.add_argument("--is_MAM", action='store_true')
parser.add_argument("--is_CSI", action='store_true')
parser.add_argument("--is_ICD", action='store_true')
parser.add_argument("--is_CSD", action='store_true')
parser.add_argument("--is_std_norm", action='store_true')
parser.add_argument("--is_init_res", action='store_true')
parser.add_argument("--is_init_he", action='store_true')

# Training / Test / Restore 
parser.add_argument("--is_resume", action='store_true')
parser.add_argument("--is_train", action='store_true')

# Training specification
parser.add_argument("--init_lr", default=0.0001, type=int)
parser.add_argument("--decay_step", default=200000, type=int)
parser.add_argument("--decay_ratio", default=0.5, type=int)
parser.add_argument("--scale", default='4')
parser.add_argument("--num_iter", default=1000000, type=int)
parser.add_argument("--num_train", default=800, type=int)
parser.add_argument("--num_batch", default=16, type=int)
parser.add_argument("--num_valid", default=100, type=int)
parser.add_argument("--num_test", default=100, type=int)

# Saver & Logger specifications
parser.add_argument("--print_freq", default=10, type=int)
parser.add_argument("--log_freq", default=100, type=int)
parser.add_argument("--save_freq", default=50000, type=int)
parser.add_argument("--valid_freq", default=200000, type=int)
parser.add_argument("--max_to_keep", default=1000000, type=int)

# Test setting
parser.add_argument("--self_ensemble", action='store_true') 
parser.add_argument("--is_test", action='store_true')
parser.add_argument("--ckpt_name", default=None)
parser.add_argument("--dataset_name", default='Set5')
	
args = parser.parse_args()