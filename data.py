import numpy as np
import tensorflow as tf
import csv


class Inputdata:

    def __init__(self, train_filename, validation_filename, test_filename):
        self.train_filename = train_filename
        self.validation_filename = validation_filename
        self.test_filename = test_filename
        self._get_train_test_data()
        self.num_of_train = np.shape(self.y_train)[0]
        self.num_of_val = np.shape(self.y_val)[0]
        self.num_of_test = np.shape(self.y_test)[0]

    def _get_train_test_data(self):
        train_data = np.loadtxt(self.train_filename, dtype=np.float32, delimiter=',')
        val_data = np.loadtxt(self.validation_filename, dtype=np.float32, delimiter=',')
        test_data = np.loadtxt(self.test_filename, dtype=np.float32, delimiter=',')

        self.x_train = train_data[:, 0:-1]
        self.y_train = train_data[:, [-1]]

        self.x_val = val_data[:, 0:-1]
        self.y_val = val_data[:, [-1]]

        self.x_test = test_data[:, 0:-1]
        self.y_test = test_data[:, [-1]]

    def get_minibatch(self, mini_batch_size):
        data_set = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        data_set = data_set.shuffle(np.shape(self.x_train)[0]).repeat().batch(mini_batch_size)
        iterator = data_set.make_one_shot_iterator()
        x, y = iterator.get_next()

        return x, y
