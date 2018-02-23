import os
import keras

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

from activations.ReLUs import ReLUs
from model.mnist_classifier import MNISTClassifier


class MNISTCNNClassifier(MNISTClassifier):

    def __init__(self, output_dir):
        super().__init__(output_dir)

    def init(self, hyper_parameters):
        super().init(hyper_parameters)

        self.dropout = hyper_parameters['dropout']
        self.e_param = hyper_parameters['e_param']
        self.activation_fn = hyper_parameters['activation_fn']

        ReLUs.config(self.e_param)

    def build_model(self):
        model = Sequential()

        model.add(Conv2D(28, 7, padding='same', activation=self.activation_fn, input_shape=(28, 28, 1)))
        model.add(Conv2D(28, 7, padding='same', activation=self.activation_fn))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.dropout))

        model.add(Conv2D(32, 3, padding='same', activation=self.activation_fn))
        model.add(Conv2D(32, 3, padding='same', activation=self.activation_fn))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.dropout))

        model.add(Conv2D(44, 2, padding='same', activation=self.activation_fn))
        model.add(Conv2D(44, 2, padding='same', activation=self.activation_fn))
        model.add(Conv2D(48, 2, activation=self.activation_fn))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.dropout))

        model.add(Flatten())
        model.add(Dense(1024, activation=self.activation_fn))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.n_classes, activation="softmax"))

        print(model.summary())
        return model

    def compile_model(self):
        model = self.build_model()

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        modelCheckpoint = ModelCheckpoint(monitor='val_acc', filepath=self.output_dir + '/weights-cnn-mnist.hdf5',
                                               save_best_only=True, mode='max')
        earlyStopping = EarlyStopping(monitor='val_acc', mode='max', patience=self.patience)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        return modelCheckpoint, earlyStopping, model

    def train_model(self):
        X_train, y_train, X_val, y_val, X_test, y_test = self.preprocess_data()

        train_datagen = ImageDataGenerator(zoom_range=0.1,
                                           height_shift_range=0.1,
                                           width_shift_range=0.1,
                                           rotation_range=10)

        test_datagen = ImageDataGenerator(zoom_range=0.1,
                                          height_shift_range=0.1,
                                          width_shift_range=0.1,
                                          rotation_range=10)

        train_generator = train_datagen.flow(X_train, y_train, batch_size=self.batch_size)
        validation_generator = test_datagen.flow(X_val, y_val, batch_size=self.batch_size)

        modelCheckpoint, earlyStopping, model = self.compile_model()

        model.fit_generator(train_generator,
                                   steps_per_epoch=self.steps_per_epoch,
                                   epochs=self.epochs,
                                   verbose=2,
                                   validation_data=validation_generator,
                                   validation_steps=self.validation_steps,
                                   callbacks=[modelCheckpoint, earlyStopping])

        # Save the current model.
        model.save(filepath=self.output_dir + '/model-cnn-mnist.hdf5')

        # Load the best weights.
        saved_model = keras.models.load_model(filepath=self.output_dir + '/weights-cnn-mnist.hdf5')

        final_loss, final_acc = saved_model.evaluate(X_test, y_test, verbose=1)
        print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))

        y_hat = saved_model.predict(self.X_test_sub, verbose=2)
        y_hat = np.argmax(y_hat, axis=1)
        self.save_submission(y_hat)

    def save_submission(self, y_hat):
        pd.DataFrame({"ImageId": list(range(1, len(y_hat) + 1)), "Label": y_hat})\
            .to_csv(self.output_dir + '/submission_cnn_mnist.csv', index=False, header=True)