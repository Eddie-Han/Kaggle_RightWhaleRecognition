import argparse

arg_lists = []
parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true', '1')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# UI
ui_arg = add_argument_group('UI')
ui_arg.add_argument('--operation_mode', type=str, default='Train', choices = ['Train', 'Test'],
                     help='The mode which you want to operate. (Train or Test)')
ui_arg.add_argument('--train_mode', type=str, default='Localize', choices = ['Localize'],
                     help='The list of the training [Localize/Align/Classify]')
ui_arg.add_argument('--test_mode', type=str, default='Localize', choices = ['Localize'],
                     help='The list of the test [Localize/Align/Classify]')

# Load Data
data_loader_arg = add_argument_group('data_loader')
data_loader_arg.add_argument('--localize_base_path', type=str,
                         default='G:\\Dataset\\NOAA_Right_Whale_Recognition\\imgs',
                         help='TODO')
data_loader_arg.add_argument('--localize_label_path', type=str,
                         default='D:\\Source\\Git\\Kaggle_RightWhaleRecognition_DeepSense\\data\\slot.json',
                         help='TODO')

# Train Common
train_common_arg = add_argument_group('train_common')


# Localize
localize_hyper_param_arg = add_argument_group('localize_hyper_param')
localize_hyper_param_arg.add_argument('--localize_ensemble_size', type=int, default=6,
                                      help='TODO')
localize_hyper_param_arg.add_argument('--localize_base_lr', type=float, default=1e-5,
                                      help='TODO')
localize_hyper_param_arg.add_argument('--localize_minibatch_size', type=int, default=32,
                                      help='TODO')
localize_hyper_param_arg.add_argument('--localize_image_resize_w', type=int, default=256,
                                      help='TODO')
localize_hyper_param_arg.add_argument('--localize_image_resize_h', type=int, default=256,
                                      help='TODO')
localize_hyper_param_arg.add_argument('--localize_validation_image_ratio', type=float, default=0.02,
                                      help='TODO')

log_arg = add_argument_group('logging')
log_arg.add_argument('--localizer_log', type=str, default = './localizer_log',
                     help = 'TODO')
log_arg.add_argument('--localizer_checkpoint', type=str, default = '/localizer_checkpoint',
                     help = 'TODO')
log_arg.add_argument('--localizer_result', type=str, default = '/localizer_result',
                     help = 'TODO')
log_arg.add_argument('--checkpoint_repository', type=str, default = './checkpoint_repository',
                     help = 'TODO')
# etc
etc_arg = add_argument_group('etc')
etc_arg.add_argument('--log_gpu_info', type=str2bool, default=False,
                     help='The boolean value about logging gpu info')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

