import numpy as np
import tensorflow as tf
import gzip
import pickle


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
        with gzip.open(self.train_filename, 'rb') as f:
            train_data = pickle.load(f)
        with gzip.open(self.validation_filename, 'rb') as f:
            val_data = pickle.load(f)
        with gzip.open(self.test_filename, 'rb') as f:
            test_data = pickle.load(f)

        self.x_train = train_data[:, 0:-1]
        self.y_train = train_data[:, [-1]]

        self.x_val = val_data[:, 0:-1]
        self.y_val = val_data[:, [-1]]

        self.x_test = test_data[:, 0:-1]
        self.y_test = test_data[:, [-1]]

    def get_minibatch(self, mini_batch_size):
        self.data_x, self.data_y = tf.placeholder(tf.float32), tf.placeholder(tf.float32)
        data_set = tf.data.Dataset.from_tensor_slices((self.data_x, self.data_y))
        data_set = data_set.shuffle(np.shape(self.x_train)[0]).repeat().batch(mini_batch_size)
        self.itr = data_set.make_initializable_iterator()
        x, y = self.itr.get_next()

        return x, y
