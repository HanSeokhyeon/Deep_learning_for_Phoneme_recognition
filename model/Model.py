import tensorflow as tf


class Model:

    """Network Model Class

    Note that this class has only the constructor.
    The actual model is defined inside the constructor.

    Attributes
    ----------
    X : tf.float32
        This is a tensorflow placeholder for TIMIT features
        Expected shape is [None, n_feature]

    y : tf.int32
        This is a tensorflow placeholder for TIMIT labels (not one hot encoded)
        Expected shape is [None, 1}

    keep_prob : tf.float32
        This is used for the dropout
        It's '0.5~0.8' at training time and '1.0' at test time

    hypothesis : tf.float32
        This is a output activation value for predict

    loss : tf.float32
        The loss function is a softmax cross entropy

    train_op
        This is simply the training op that minimizes the loss

    accuracy : tf.float32
        The accuracy operation


    Examples
    --------
    #>>> model = Model("PhonemeNN", 96, 39)
    """

    def __init__(self,
                 name,
                 input_dim, output_dim, hidden_dims=[32, 32],
                 activation_fn=tf.nn.relu,
                 optimizer=tf.train.AdamOptimizer,
                 lr=0.001):
        """ Constructor

        :param name: str
                    The name of this network
                    The entire network will be created under 'tf.variable_scope(name)'

        :param input_dim: int
                    The input dimension
                    In this example, 784

        :param output_dim: int
                    The number of output labels
                    There are 10 labels

        :param hidden_dims: list (default: [DNN32, DNN32])
                    len(hidden_dims) = number of layers
                    each element is the number of hidden units

        :param activation_fn: Tf functions (default: tf.nn.relu)
                    Activation Function

        :param optimizer: TF optimizer (default: tf.train.AdamOptimizer)
                    Optimizer Function

        :param lr: float (default : 0.01)
                    Learning rate
        """

        with tf.variable_scope(name):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

            # Placeholders are defined
            self.X = tf.placeholder(tf.float32, [None, input_dim], name='X')
            self.y = tf.placeholder(tf.int32, [None, 1], name='y')

            y_one_hot = tf.one_hot(self.y, output_dim)
            y_one_hot = tf.reshape(y_one_hot, [-1, output_dim])

            self.keep_prob = tf.placeholder(tf.float32)

            # Loop over hidden layers
            net = self.X
            for i, h_dim in enumerate(hidden_dims):
                with tf.variable_scope('layer{}'.format(i)):
                    net = tf.layers.dense(net, h_dim)
                    net = activation_fn(net)
                    net = tf.nn.dropout(net, keep_prob=self.keep_prob)

            # Attach fully connected layers
            net = tf.contrib.layers.flatten(net)
            net = tf.layers.dense(net, output_dim)
            self.hypothesis = net

            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y_one_hot)
            self.loss = tf.reduce_mean(self.loss, name='loss')

            self.train_op = optimizer(lr).minimize(self.loss, global_step=self.global_step)

            # Accuracy etc
            softmax = tf.nn.softmax(net, name='softmax')
            self.accuracy = tf.equal(tf.argmax(softmax, 1), tf.argmax(y_one_hot, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))

            self.saver = tf.train.Saver(tf.global_variables())
