import os as os
import tensorflow as tf
import numpy as np
import time
import sys
from util import *
from config import *
from data_loader import *
from localizer import *

def main(config):
   if config.operation_mode == 'Train':
       train(config)
   elif config.operation_mode == 'Test':
        print('Test')
   else:
       assert False, 'Invalid operation mode : {}'.format(config.operation_mode)

def train(config):

    if not os.path.exists(config.localizer_log):
        os.makedirs(config.localizer_log)
    if not os.path.exists(config.localizer_log + config.localizer_checkpoint):
        os.makedirs(config.localizer_log + config.localizer_checkpoint)
    if not os.path.exists(config.localizer_log + config.localizer_result):
        os.makedirs(config.localizer_log + config.localizer_result)
    if not os.path.exists(config.checkpoint_repository):
        os.makedirs(config.checkpoint_repository)

    tensor_config = tf.ConfigProto(log_device_placement=config.log_gpu_info,
                                   device_count={'GPU': 1})
    tensor_config.gpu_options.allow_growth = True

    train_mode_list = parse_train_mode(config.train_mode)
    if 'Localize' in train_mode_list:
        train_localize(config, tensor_config)
    if 'Align' in train_mode_list:
        print('[Not Implemented] Align - Train')
    if 'Classify' in train_mode_list:
        print('[Not Implemented] Classify - Train')

def train_localize(config, tensor_config):
    #init data loader
    data_loader = DataLoader(config, 'train_localize',
                                  config.localize_image_resize_w, config.localize_image_resize_h)
    with tf.Session(config=tensor_config) as sess:
        localizer = Localizer(
            sess = sess,
            data_loader = data_loader,
            ensemble_size=  config.localize_ensemble_size,
            input_size=(config.localize_image_resize_w, config.localize_image_resize_h),
            minibatch_size=config.localize_minibatch_size,
            base_lr=config.localize_base_lr,
            log_path=config.localizer_log,
            checkpoint_path=config.localizer_checkpoint
        )
        localizer.do_train()

if __name__ == "__main__":
    config, unparsed = get_config()
    sys.exit(main(config))