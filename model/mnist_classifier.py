import keras
import numpy as np

from sklearn.model_selection import train_test_split

np.random.seed(42)


class MNISTClassifier:

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def init(self, hyper_parameters):
        self.n_classes = hyper_parameters['n_classes']

        self.epochs = hyper_parameters['epochs']
        self.steps_per_epoch = hyper_parameters['steps_per_epoch']
        self.validation_steps = hyper_parameters['validation_steps']
        self.batch_size = hyper_parameters['batch_size']
        self.patience = hyper_parameters['patience']
        self.test_split = hyper_parameters['test_split']

        self.load_data()

    def load_data(self):
        self.X_train_all = np.loadtxt('data/mnist/train.csv', skiprows=1, dtype='int', delimiter=',')
        self.X_test_sub = np.loadtxt('data/mnist/test.csv', skiprows=1, dtype='int', delimiter=',')
        self.X_test_sub = self.X_test_sub.reshape(28000, 28, 28, 1).astype('float32') / 255.

    def preprocess_data(self):
        X_train, X_val, y_train, y_val = train_test_split(self.X_train_all[:, 1:], self.X_train_all[:, 0], test_size=self.test_split)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=self.test_split)

        X_train = X_train.reshape(-1, 28, 28, 1)
        X_val = X_val.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)

        X_train = X_train.astype("float32") / 255.
        X_val = X_val.astype("float32") / 255.
        X_test = X_test.astype("float32") / 255.

        y_train = keras.utils.to_categorical(y_train, self.n_classes)
        y_val = keras.utils.to_categorical(y_val, self.n_classes)
        y_test = keras.utils.to_categorical(y_test, self.n_classes)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def build_model(self):
        pass

    def compile_model(self):
        pass

    def train_model(self):
        pass

    def save_submission(self, y_hat):
        pass