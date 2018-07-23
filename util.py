from define import *

def parse_train_mode(train_mode):
    train_mode_list = train_mode.split('/')
    available_train_mode_list = get_available_train_mode()

    for train_mode_elem in train_mode_list:
            if train_mode_elem not in available_train_mode_list:
                assert False, 'Invalid train mode : {}'.format(train_mode_elem)

    return train_mode_list

