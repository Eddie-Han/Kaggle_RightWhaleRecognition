import json
import os
import numpy as np
from util_image import *
#from config import *

class DataLoader:
    def __init__(self, config, mode, resize_w, resize_h):
        self.mode = mode
        self.batch_idx = 0
        self.resize_w = resize_w
        self.resize_h = resize_h

        if mode == 'train_localize':
            self.train_localize_init(config)
        else:
            assert False, 'Invalid data loader mode : {}'.format(mode)

    def reset_batch(self):
        self.batch_idx = 0

    def next_batch(self):
        if self.mode == 'train_localize':
            images_path = [self.localize_base_path + '\\' + self. label_data[idx]['filename']
                for idx in range(self.batch_idx, self.batch_idx + self.minibatch_size)]
            images = [get_image(image_path, self.resize_w, self.resize_h, True)
                      for image_path in images_path]
            images = np.array(images).astype(np.float32) #TODO::epoch이 반복될때는 어떻게 처리?
            return images
        else:
            assert False, 'Invalid data loader mode : {}'.format(self.mode)

    def train_localize_init(self, config, do_shuffle = True):
        self.minibatch_size = config.localize_minibatch_size
        self.localize_base_path = config.localize_base_path

        # init label data
        label_path = config.localize_label_path
        if not os.path.exists(label_path):
            assert False, 'Invalid label path : {}'.format(label_path)
        with open(label_path) as data_file:
            self.label_data = json.load(data_file)
            if do_shuffle == True :
                np.random.shuffle(self.label_data)

    def resize_value(self):
        #TODO::resize 관련 data augumentation logic이 있으면 여기서 처리한다.
        return self.resize_w, self.resize_h

    def reshuffle(self): # for ensemble logic
        np.random.shuffle(self.label_data)
