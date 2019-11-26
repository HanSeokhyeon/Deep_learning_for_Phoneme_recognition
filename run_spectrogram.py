from Model import *
from Solver import *
from data import *
from feature.spectrogram_multiprocessing import make_spectrogram_feature
from logger import *

import time
import math
import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

input_dim = 120
hidden_dims = [2000, 1000, 1000]
output_dim = 39

feature_name = 'spectrogram'
data_name = ['TRAIN', 'TEST_developmentset', 'TEST_coreset']

epoch_n = 100
batch_size = 8192
dropout_rate = 0.8
patience = 10


def main():
    if not os.path.isfile('input/%d_%s_%s.pickle' % (input_dim, feature_name, data_name[0])) or \
        not os.path.isfile('input/%d_%s_%s.pickle' % (input_dim, feature_name, data_name[1])) or \
        not os.path.isfile('input/%d_%s_%s.pickle' % (input_dim, feature_name, data_name[2])):
        make_spectrogram_feature()

    inputdata = Inputdata('input/%d_%s_%s.pickle' % (input_dim, feature_name, data_name[0]),
                          'input/%d_%s_%s.pickle' % (input_dim, feature_name, data_name[1]),
                          'input/%d_%s_%s.pickle' % (input_dim, feature_name, data_name[2]))

    logger.info("data load complete")

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    x, y = inputdata.get_minibatch(batch_size)

    nn = Model('nn',
               input_dim=input_dim,
               output_dim=output_dim,
               hidden_dims=hidden_dims)

    # We create two solvers: to train both models at the same time for comparison
    # Usually we only need one solver class
    nn_solver = Solver(sess, nn)

    init = tf.global_variables_initializer()
    sess.run(init)

    logger.info(input_dim, feature_name)

    min_loss = math.inf
    patience_count = 0

    for epoch in range(epoch_n):
        start = time.time()
        train_loss = 0

        for _ in range(inputdata.num_of_train//batch_size):
            X_batch, y_batch = sess.run([x, y])

            _, nn_loss = nn_solver.train(X_batch, y_batch, dropout_rate)
            train_loss += nn_loss

        train_loss /= inputdata.num_of_train//batch_size
        n_loss, n_acc = nn_solver.evaluate(inputdata.x_val, inputdata.y_val)

        end = time.time() - start

        logger.info('Epoch', '%04d' % (epoch + 1),
                    '  Train loss =', '{:.9f}'.format(train_loss),
                    '  Val loss =', '{:.9f}'.format(n_loss),
                    '  Val accuracy = %.4f' % n_acc,
                    '  patience = %d' % patience_count,
                    '  time = %.1f' % end)

        if n_loss < min_loss:
            min_loss = n_loss
            nn.saver.save(sess, 'model/dnn.ckpt')
            patience_count = 0
        else:
            patience_count += 1
            if patience_count == patience:
                logger.info('Early Stopping!')
                break

    nn.saver.restore(sess, tf.train.latest_checkpoint('model'))
    val_loss, val_acc = nn_solver.evaluate(inputdata.x_val, inputdata.y_val)
    test_loss, test_acc = nn_solver.evaluate(inputdata.x_test, inputdata.y_test)
    logger.info('Val loss =', '{:.9f}'.format(val_loss),
                '  Val accuracy = %.4f' % val_acc,
                '  Test loss =', '{:.9f}'.format(test_loss),
                '  Test accuracy = %.4f' % test_acc)

    y_ = np.argmax(nn_solver.predict(inputdata.x_test), axis=1)
    y_label = np.transpose(inputdata.y_test)[0]
    cm = tf.confusion_matrix(labels=y_label, predictions=y_, num_classes=output_dim)
    now_cm = cm.eval()
    np.savetxt('output/%d_%s_CM.csv' % (input_dim, feature_name), now_cm, delimiter=',')

    return 0


if __name__ == '__main__':
    main()
