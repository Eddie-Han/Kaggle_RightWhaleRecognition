import json
import os
import numpy as np
from util_image import *
#from config import *

class DataLoader:
    def __init__(self, config, mode, resize_w, resize_h):
        self.mode = mode
        self.batch_idx = 0
        self.batch_idx_test = 0
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.epoch_cnt = 0


        if mode == 'train_localize':
            self.train_localize_init(config)
        else:
            assert False, 'Invalid data loader mode : {}'.format(mode)

    def reset_batch(self):
        self.batch_idx = 0

    def next_batch(self, do_normalize = True):
        if self.mode == 'train_localize':
            if self.batch_idx + self.minibatch_size > len(self.label_data):
                self.epoch_cnt += 1
                self.reset_batch()

            images_path = []
            images_info =[]
            for idx in range(self.batch_idx, self.batch_idx + self.minibatch_size):
                images_path.append(self.localize_base_path + '\\' + self. label_data[idx]['filename'])
                entire_image_info = get_image_info(self.localize_base_path + '\\' + self.label_data[idx]['filename'])
                label_info = self.label_data[idx]['annotations'][0]
                if do_normalize == True:
                    image_info = {'image_height': entire_image_info[0],
                                  'image_width': entire_image_info[1],
                                  'image_channel': entire_image_info[2],
                                  'label_x': label_info['x'] / entire_image_info[1],
                                  'label_y': label_info['y'] / entire_image_info[0],
                                  'label_width': label_info['width'] / entire_image_info[1],
                                  'label_height': label_info['height'] / entire_image_info[0]}
                else:
                    image_info = {'image_height': entire_image_info[0], 'image_width': entire_image_info[1],
                                  'image_channel': entire_image_info[2], 'label_x': label_info['x'],
                                  'label_y': label_info['y'], 'label_width': label_info['width'],
                                  'label_height': label_info['height']}
                images_info.append(image_info)

            self.batch_idx += self.minibatch_size
            images = [get_image(image_path, self.resize_w, self.resize_h, True)
                      for image_path in images_path]
            images = np.array(images).astype(np.float32)

            return images, images_info
        else:
            assert False, 'Invalid data loader mode : {}'.format(self.mode)

    def next_batch_test(self, do_normalize = True):
        if self.mode == 'train_localize':
            if self.batch_idx_test + self.minibatch_size > len(self.label_data_test):
                self.batch_idx_test = 0
                return [], []

            images_path = []
            images_info =[]
            for idx in range(self.batch_idx_test, self.batch_idx_test + self.minibatch_size):
                images_path.append(self.localize_base_path + '\\' + self. label_data_test[idx]['filename'])
                entire_image_info = get_image_info(self.localize_base_path + '\\' + self.label_data_test[idx]['filename'])
                label_info = self.label_data_test[idx]['annotations'][0]
                if do_normalize == True:
                    image_info = {'image_height': entire_image_info[0],
                                  'image_width': entire_image_info[1],
                                  'image_channel': entire_image_info[2],
                                  'label_x': label_info['x'] / entire_image_info[1],
                                  'label_y': label_info['y'] / entire_image_info[0],
                                  'label_width': label_info['width'] / entire_image_info[1],
                                  'label_height': label_info['height'] / entire_image_info[0]}
                else:
                    image_info = {'image_height': entire_image_info[0], 'image_width': entire_image_info[1],
                                  'image_channel': entire_image_info[2], 'label_x': label_info['x'],
                                  'label_y': label_info['y'], 'label_width': label_info['width'],
                                  'label_height': label_info['height']}
                images_info.append(image_info)

            self.batch_idx_test += self.minibatch_size
            images = [get_image(image_path, self.resize_w, self.resize_h, True)
                      for image_path in images_path]
            images = np.array(images).astype(np.float32)

            return images, images_info
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

            train_test_sep = int(len(self.label_data) * config.localize_validation_image_ratio)
            label_data_train = self.label_data[train_test_sep : ]
            self.label_data_test = self.label_data[0 : train_test_sep]
            self.label_data = label_data_train

    def resize_value(self):
        #TODO::resize 관련 data augumentation logic이 있으면 여기서 처리한다.
        return self.resize_w, self.resize_h

    def reshuffle(self): # for ensemble logic
        np.random.shuffle(self.label_data)

    def current_epoch(self):
        return self.epoch_cnt
