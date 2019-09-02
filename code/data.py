import random, os, imageio
import numpy as np
from utils import mod_crop

class DataGenerator():
    def __init__(self, args):
        self.args = args
        self.load_data()
        print('Data is loaded!')
    
    # load all dataset (HR and LR used in an experiment)
    def load_data(self):
        # initialze dataset
        self.dataset = []
        # check scales used in an experiment 
        self.scale_list = list(map(lambda x: int(x), self.args.scale.split('+')))

        if self.args.is_test:
            dirs = [os.path.join(self.args.data_dir, "LR", "x%d" % self.scale_list[0], self.args.dataset_name)]
            dirs.append(os.path.join(self.args.data_dir, "HR", self.args.dataset_name))

            # The list of file_names for each directory 
            self.file_names_for_dirs = [sorted([f for f in os.listdir(dir) if not f=='Thumbs.db']) for dir in dirs]

            # Load data for LR 
            for dir, file_names in zip(dirs, self.file_names_for_dirs):
                tmp = []
                for file_name in file_names:
                    tmp.append(imageio.imread(dir + "/" + file_name, pilmode="RGB"))
                self.dataset.append(tmp)    
        else:
            # directories for LR (for scale used) and HR (ex. [x2, x4, x8, HR])
            dirs = [os.path.join(self.args.data_dir, "DIV2K_train_LR_bicubic/X%d" % scale) for scale in self.scale_list]
            dirs.append(os.path.join(self.args.data_dir, "DIV2K_train_HR"))

            # The list of file_names for each directory 
            self.file_names_for_dirs = [sorted([f for f in os.listdir(dir) if not f=='Thumbs.db'])[:self.args.num_train] for dir in dirs]

            # Load data for LR and HR (ex. self.dataset = [[x2_data], [x4_data], [x8_data], [HR_data]])
            for dir, file_names in zip(dirs, self.file_names_for_dirs):
                tmp = []
                for file_name in file_names:
                    tmp.append(imageio.imread(dir + "/" + file_name))
                self.dataset.append(tmp)

    # construct batch data for randomly selected scale
    # only use this function during traing, not validation, not testing 
    def get_batch(self, batch_size, idx_scale, in_patch_size=48):
        # randomly selet scale
        scale = self.scale_list[idx_scale]

        # assgin target patch size
        tar_patch_size = in_patch_size * scale
        in_batch = []
        tar_batch = []
        for _ in range(batch_size):
            # select image
            idx_img = random.randrange(self.args.num_train)
            in_img = self.dataset[idx_scale][idx_img]
            tar_img = self.dataset[-1][idx_img]

            # random crop
            y, x, _ = in_img.shape
            in_x = random.randint(0, x - in_patch_size)
            in_y = random.randint(0, y - in_patch_size) 
            tar_x = in_x * scale
            tar_y = in_y * scale 
            in_patch = in_img[in_y:in_y + in_patch_size, in_x:in_x + in_patch_size]  
            tar_patch = tar_img[tar_y:tar_y + tar_patch_size, tar_x:tar_x + tar_patch_size]  

            # random rotate
            rot_num = random.randint(1, 4)
            in_patch = np.rot90(in_patch, rot_num)
            tar_patch = np.rot90(tar_patch, rot_num)
            
            # random flip left-to-right
            flipflag = random.random() > 0.5
            if flipflag:
                in_patch = np.fliplr(in_patch) 
                tar_patch = np.fliplr(tar_patch)

            # construct mini-batch
            in_batch.append(in_patch)
            tar_batch.append(tar_patch)                              
        return in_batch, tar_batch
