from Model import *
from Solver import *
from data import *
import tensorflow as tf
import time
import csv
import math


input_dim = 126
hidden_dims = [2000, 1000, 1000]
output_dim = 39

feature_name = 'PSNR50'
data_name = ['TRAIN', 'TEST_developmentset', 'TEST_coreset']

epoch_n = 100
batch_size = 8192
dropout_rate = 0.8
patience = 5


def main():
    inputdata = Inputdata('feature/%d_%s_%s.csv' % (input_dim, feature_name, data_name[0]),
                          'feature/%d_%s_%s.csv' % (input_dim, feature_name, data_name[1]),
                          'feature/%d_%s_%s.csv' % (input_dim, feature_name, data_name[2]))

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

    ckpt = tf.train.get_checkpoint_state('./model')

    init = tf.global_variables_initializer()
    sess.run(init)

    print(input_dim, feature_name)

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

        print('Epoch', '%04d' % (epoch + 1),
              '  Train loss =', '{:.9f}'.format(train_loss),
              '  Val loss =', '{:.9f}'.format(n_loss),
              '  Val accuracy = %.4f' % n_acc,
              '  patience = %d' % patience_count,
              '  time = %.1f' % end)

        if n_loss < min_loss:
            min_loss = n_loss
            nn.saver.save(sess, './model/dnn.ckpt')
            patience_count = 0
        else:
            patience_count += 1
            if patience_count == patience:
                print('Early Stopping!')
                break

    nn.saver.restore(sess, ckpt.model_checkpoint_path)
    val_loss, val_acc = nn_solver.evaluate(inputdata.x_val, inputdata.y_val)
    test_loss, test_acc = nn_solver.evaluate(inputdata.x_test, inputdata.y_test)
    print('Val loss =', '{:.9f}'.format(val_loss),
          '  Val accuracy = %.4f' % val_acc,
          '  Test loss =', '{:.9f}'.format(test_loss),
          '  Test accuracy = %.4f' % test_acc)

    y_ = np.argmax(nn_solver.predict(inputdata.x_test), axis=1)
    y_label = np.transpose(inputdata.y_test)[0]
    cm = tf.confusion_matrix(labels=y_label, predictions=y_, num_classes=output_dim)
    now_cm = cm.eval()
    save_confusion_matrix(now_cm)


def save_confusion_matrix(matrix):
    output_filename = 'output/%d_%s_CM.csv' % (input_dim, feature_name)
    fo1 = open(output_filename, 'w', encoding='utf-8', newline='')
    writer1 = csv.writer(fo1)
    writer1.writerows(matrix)
    fo1.close()

    return


if __name__ == '__main__':
    main()
