class Solver:
    """Solver class

    This class will contain the model class and session

    Attributes
    ----------
    model : Model class
    sess : TF session

    Methods
    ----------
    train(X, y)
        Run the train_op and Returns the loss

    evaluate(X, y, batch_size=None)
        Returns "Loss" and "Accuracy"
        If batch_size is given, it's computed using batch_size
        because most GPU memories cannot handle the entire training data at once

    Example
    ----------
    #>>> sess = tf.InteractiveSession()
    #>>> model = Model("BatchNorm", DNN32, 10)
    #>>> solver = Solver(sess, model)

    # Train
    #>>> solver.train(X, y)

    # Evaluate
    #>>> solver.evaluate(X, y)
    """

    def __init__(self, sess, model):
        self.model = model
        self.sess = sess

    def train(self, X, y, keep_prob):
        feed = {
            self.model.X: X,
            self.model.y: y,
            self.model.keep_prob: keep_prob
        }
        train_op = self.model.train_op
        loss = self.model.loss

        return self.sess.run([train_op, loss], feed_dict=feed)

    def evaluate(self, X, y, batch_size=None, keep_prob=1.0):
        if batch_size:
            N = X.shape[0]

            total_loss = 0
            total_acc = 0

            for i in range(0, N, batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                feed = {
                    self.model.X: X_batch,
                    self.model.y: y_batch,
                    self.model.keep_prob: keep_prob
                }

                loss = self.model.loss
                accuracy = self.model.accuracy

                step_loss, step_acc = self.sess.run([loss, accuracy], feed_dict=feed)

                total_loss += step_loss * X_batch.shape[0]
                total_acc += step_acc * X_batch.shape[0]

            total_loss /= N
            total_acc /= N

            return total_loss, total_acc

        else:
            feed = {
                self.model.X: X,
                self.model.y: y,
                self.model.keep_prob: keep_prob
            }

            loss = self.model.loss
            accuracy = self.model.accuracy

            return self.sess.run([loss, accuracy], feed_dict=feed)

    def predict(self, X, keep_prob=1.0):
        feed = {
            self.model.X: X,
            self.model.keep_prob: keep_prob
        }

        hypothesis = self.model.hypothesis

        return self.sess.run(hypothesis, feed_dict=feed)
