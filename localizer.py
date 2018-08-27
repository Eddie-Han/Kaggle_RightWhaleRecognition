import tensorflow as tf
from tensor_util import *
from data_loader import *

class Localizer:
    def __init__(self, sess, data_loader, ensemble_size, input_size, minibatch_size, base_lr, log_path, checkpoint_path):
        self.sess = sess
        self.ensemble_size = ensemble_size
        self.data_loader = data_loader
        self.input_size = input_size
        self.minibatch_size = minibatch_size
        self.base_lr = base_lr
        self.log_path = log_path
        self.checkpoint_path = checkpoint_path

    def do_train(self):
        for iter in range(self.ensemble_size-1):
            self._do_train_iter(iter)
            # data_loader reset 하는 과정 포함되야 한다. 다음 network train 할때 epoch이 유지되는 현상 있음.

    def _do_train_iter(self, ensemble_net_iter):
        ensemble_net_name = 'LocalizeNet-{}'.format(ensemble_net_iter)
        localizer_net = self._build_network(ensemble_net_name)
        check_point_saver = tf.train.Saver(max_to_keep=1)
        tf.global_variables_initializer().run()

        BATCH_MAX = 3000
        TRAIN_LOGGING_STEP = 5
        TEST_PERIOD = 50
        SAVE_MODEL_PERIOD = 1000
        for train_step in range(0, BATCH_MAX):
            batch_images, batch_images_info = self.data_loader.next_batch(do_normalize=True)
            y_actual_input = [[batch_image_info['label_x'], batch_image_info['label_y'],
                               batch_image_info['label_width'], batch_image_info['label_height']]
                for batch_image_info in batch_images_info]
            _, loss_value = self.sess.run([localizer_net.optimizer, localizer_net.loss]
                                          , feed_dict={localizer_net.input_tensor: batch_images,
                                                       localizer_net.y_actual: y_actual_input,
                                                       localizer_net.is_training:True})
            if train_step % TRAIN_LOGGING_STEP == 0:
                debug_str = str('[' + ensemble_net_name + ']' + 'epoch : {0:>03} train_step = {1:>05}, loss = {2:>05} \n'
                    .format(self.data_loader.current_epoch(), train_step, loss_value))
                print(debug_str)
                train_result_debug_file = open(
                    self.log_path  + '/' + ensemble_net_name + '_result_debug.txt', 'a')
                train_result_debug_file.write(debug_str)
                train_result_debug_file.close()

            if train_step % TEST_PERIOD == 0 :
                self._do_train_test(localizer_net, ensemble_net_name, train_step)

            if train_step % SAVE_MODEL_PERIOD == 0:
                checkpoint_save(sess= self.sess, saver = check_point_saver,
                                checkpoint_dir=self.log_path + self.checkpoint_path + '/' + ensemble_net_name)
                print(str('Save checkpoint - {}'.format(ensemble_net_name)))

    def _do_train_test(self, net, net_name, train_step):
        loss_sum = 0.
        for test_step in range(0, 9999):
            batch_images, batch_images_info = self.data_loader.next_batch_test(do_normalize=True)
            if len(batch_images) == 0:
                debug_str = str('[TEST_RESULT][' + net_name + ']' + 'epoch : {0:>03} train_step = {1:>05}, loss = {2:>05} \n'
                          .format(self.data_loader.current_epoch(), train_step, loss_sum / test_step))
                print(debug_str)
                train_result_debug_file = open(
                    self.log_path + '/' + net_name + '_result_debug.txt', 'a')
                train_result_debug_file.write(debug_str)
                train_result_debug_file.close()
                break

            y_actual_input = [[batch_image_info['label_x'], batch_image_info['label_y'],
                               batch_image_info['label_width'], batch_image_info['label_height']]
                              for batch_image_info in batch_images_info]
            _, loss_value = self.sess.run([net.optimizer, net.loss]
                                          , feed_dict={net.input_tensor: batch_images,
                                                       net.y_actual: y_actual_input,
                                                       net.is_training: False})
            loss_sum +=  loss_value


    def _build_network(self, ensemble_net_name):
        print('Build network start : ' + ensemble_net_name)

        INIT_DEV = 0.02
        OUTPUT_SIZE = 4

        with tf.name_scope(ensemble_net_name):
            input_image_tensor = tf.placeholder(tf.float32, shape=self._input_image_shape(self.input_size))
            is_training = tf.placeholder(tf.bool, name='is_training')

            # layer-0
            l_bn0 = batch_norm(name=ensemble_net_name + '_bn0')
            l_relu0 = activate(activation='relu', input_tensor=l_bn0(conv2d(
                input=input_image_tensor,output_dim=16, k_w=3, k_h=3,d_w=1,d_h=1,stddev=INIT_DEV,
                name=ensemble_net_name + '_conv0'),train=is_training),name = ensemble_net_name + '_relu0')
            l_mp0 = max_pool(l_relu0, k_h=3 ,k_w=3 ,d_h=2, d_w=2, padding='SAME', name = ensemble_net_name + '_pool0')

            # layer-1
            l_bn1 = batch_norm(name=ensemble_net_name + '_bn1')
            l_relu1 = activate(activation='relu', input_tensor=l_bn1(conv2d(
                input=l_mp0, output_dim=64, k_w=3, k_h=3, d_w=1, d_h=1, stddev=INIT_DEV,
                name=ensemble_net_name + '_conv1'), train=is_training), name=ensemble_net_name + '_relu1')
            l_mp1 = max_pool(l_relu1, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME', name=ensemble_net_name + '_pool1')

            # layer-2
            l_bn2 = batch_norm(name=ensemble_net_name + '_bn2')
            l_relu2 = activate(activation='relu', input_tensor=l_bn2(conv2d(
                input=l_mp1, output_dim=64, k_w=3, k_h=3, d_w=1, d_h=1, stddev=INIT_DEV,
                name=ensemble_net_name + '_conv2'), train=is_training), name=ensemble_net_name + '_relu2')
            l_mp2 = max_pool(l_relu2, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME', name=ensemble_net_name + '_pool2')

            # layer-3
            l_bn3 = batch_norm(name=ensemble_net_name + '_bn3')
            l_relu3 = activate(activation='relu', input_tensor=l_bn3(conv2d(
                input=l_mp2, output_dim=64, k_w=3, k_h=3, d_w=1, d_h=1, stddev=INIT_DEV,
                name=ensemble_net_name + '_conv3'), train=is_training), name=ensemble_net_name + '_relu3')
            l_mp3 = max_pool(l_relu3, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME', name=ensemble_net_name + '_pool3')

            # layer-4
            l_bn4 = batch_norm(name=ensemble_net_name + '_bn4')
            l_relu4 = activate(activation='relu', input_tensor=l_bn4(conv2d(
                input=l_mp3, output_dim=64, k_w=3, k_h=3, d_w=1, d_h=1, stddev=INIT_DEV,
                name=ensemble_net_name + '_conv4'), train=is_training), name=ensemble_net_name + '_relu4')
            l_mp4 = max_pool(l_relu4, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME', name=ensemble_net_name + '_pool4')

            # Dense
            l_lin0 = linear(input= tf.reshape(l_mp4, [-1, 8*8*64]), output_size=OUTPUT_SIZE, stddev= INIT_DEV,
                            scope = ensemble_net_name + '_lin')

            # Results
            y_actual = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])
            softmax_result = tf.nn.softmax(l_lin0)

            # Get target variables
            trainable_vars = tf.trainable_variables()
            trainable_vars = [var for var in trainable_vars if ensemble_net_name in var.name]

            # Loss
            loss = loss_func('rmse', y_actual, softmax_result)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.base_lr) \
                .minimize(loss, var_list=trainable_vars)

        localizer_net = LocalizerNet(net_name= ensemble_net_name, input_tensor= input_image_tensor, is_training = is_training
                               , y_actual= y_actual, softmax_result= softmax_result, loss= loss, optimizer= optimizer)

        print('Build network end : ' + ensemble_net_name)

        return localizer_net


    def _input_image_shape(self, input_size, color_channels = 3):
        input_shape = [None, color_channels]
        input_shape.insert(1, input_size[0])
        input_shape.insert(2, input_size[1])
        return input_shape

class LocalizerNet:
    def __init__(self, net_name, input_tensor, is_training, y_actual, softmax_result, loss, optimizer):
        self.net_name = net_name
        self.input_tensor = input_tensor
        self.is_training = is_training
        self.y_actual = y_actual
        self.softmax_result = softmax_result
        self.loss = loss
        self.optimizer= optimizer