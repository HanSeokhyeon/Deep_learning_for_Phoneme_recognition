from model.Model import *
from model.Solver import *
from data import *
from feature.melspectrogram_multiprocessing import make_melspectrogram_feature
from logger import *
from result import *

import time
import math
import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

input_dim = 120
hidden_dims = [2000, 1000, 1000]
output_dim = 39

feature_name = 'melspectrogram'
data_name = ['TRAIN', 'TEST_developmentset', 'TEST_coreset']

epoch_n = 100
batch_size = 8192
dropout_rate = 0.8
patience = 10


def main():
    if not os.path.isfile('input/%d_%s_%s.pickle' % (input_dim, feature_name, data_name[0])) or \
        not os.path.isfile('input/%d_%s_%s.pickle' % (input_dim, feature_name, data_name[1])) or \
        not os.path.isfile('input/%d_%s_%s.pickle' % (input_dim, feature_name, data_name[2])):
        make_melspectrogram_feature()

    inputdata = Inputdata('input/%d_%s_%s.pickle' % (input_dim, feature_name, data_name[0]),
                          'input/%d_%s_%s.pickle' % (input_dim, feature_name, data_name[1]),
                          'input/%d_%s_%s.pickle' % (input_dim, feature_name, data_name[2]))

    logger.info("data load complete")

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    x, y = inputdata.get_minibatch(batch_size)

    sess.run(inputdata.itr.initializer, feed_dict={inputdata.data_x:inputdata.x_train, inputdata.data_y:inputdata.y_train})

    nn = Model('nn',
               input_dim=input_dim,
               output_dim=output_dim,
               hidden_dims=hidden_dims)

    # We create two solvers: to train both models at the same time for comparison
    # Usually we only need one solver class
    nn_solver = Solver(sess, nn)

    init = tf.global_variables_initializer()
    sess.run(init)

    logger.info("{} {}".format(input_dim, feature_name))

    min_loss = math.inf
    patience_count = 0

    loss_tr, loss_val, acc_val = [], [], []

    for epoch in range(epoch_n):
        start = time.time()
        train_loss = 0

        for _ in range(inputdata.num_of_train // batch_size):
            X_batch, y_batch = sess.run([x, y])

            _, nn_loss = nn_solver.train(X_batch, y_batch, dropout_rate)
            train_loss += nn_loss

        train_loss /= inputdata.num_of_train // batch_size
        loss_tr.append(train_loss)

        n_loss, n_acc = nn_solver.evaluate(inputdata.x_val, inputdata.y_val)
        loss_val.append(n_loss)
        acc_val.append(n_acc)

        end = time.time() - start

        logger.info("Epoch {} Train loss = {:.9f} Val loss = {:.9f} Val acc = {:.4f} patience = {} time = {:.4f}" \
                    .format((epoch + 1), train_loss, n_loss, n_acc, patience_count, end))

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

    logger.info("Val loss ={:.9f} Val acc ={:.4f} Test loss ={:.9f} Test acc ={:.4f}" \
                .format(val_loss, val_acc, test_loss, test_acc))

    y_ = np.argmax(nn_solver.predict(inputdata.x_test), axis=1)
    y_label = np.transpose(inputdata.y_test)[0]
    cm = tf.confusion_matrix(labels=y_label, predictions=y_, num_classes=output_dim)
    now_cm = cm.eval()
    np.savetxt('output/%d_%s_CM.csv' % (input_dim, feature_name), now_cm, delimiter=',')

    plot(loss_tr=loss_tr, loss_val=loss_val, acc_val=acc_val, acc_test=test_acc,
         filename="output/{}_{}.png".format(input_dim, feature_name))
    phonetic_accuracy(now_cm)

    return


if __name__ == '__main__':
    main()
