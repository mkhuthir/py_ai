#!/usr/bin/python3

from load_data import x_test, x_train, y_test, y_train
from keras.callbacks import ModelCheckpoint
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D)
from keras.models import Sequential, load_model
import tensorflow as tf
import matplotlib.pyplot as plt

def create_model():
    """
    Create the model.
    """
    model = Sequential()
    model.add(Conv2D(
        filters=10,
        kernel_size=(4, 4),
        input_shape=(28, 28, 1), padding='same'))

    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(
        filters=40, 
        kernel_size=(4, 4), padding='same'))
        
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))
    return model

def compile_model(model):
    """
    Compile the model.
    """
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=['categorical_accuracy'])
    return model

def fit_model(model):
    """
    Train the model while storing the best weights.
    """
    checkpoint = ModelCheckpoint(
        filepath='models/mnist.hdf5',
        monitor='categorical_accuracy',
        save_best_only=True,
        mode='max')

    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=100,
        validation_split=0.25,
        callbacks=[checkpoint])
    return model, history

def plot_history(history):
    """
    Plot model accuracy and loss histories
    """
    # plot model history
    f, ax = plt.subplots(1, 2, figsize=(12,6))

    # Plot training & validation accuracy values
    ax[0].plot(history.history['categorical_accuracy'])
    ax[0].plot(history.history['val_categorical_accuracy'])
    ax[0].set_title('Model accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('Model loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Test'], loc='upper left')

    plt.show()


def evaluate_model():
    """
    Evaluate the model.
    """
    best_model = load_model('models/mnist.hdf5')
    train_score = best_model.evaluate(x_train, y_train)
    test_score = best_model.evaluate(x_test, y_test)
    return train_score, test_score

# The following is to suppress TensorFlow Warnings
# tf.logging.set_verbosity(tf.logging.ERROR)

print('\nInitialize the model')
initial_model = create_model()

print('Compile the model')
compiled_model = compile_model(initial_model)

print('Train the model')
trained_model, history = fit_model(compiled_model)

print('Showing model training history')
# if required use print(history.history.keys()) to list all data in history
plot_history(history)

print('\nEvaluating the model...')
train_score, test_score = evaluate_model()

print('\nPercentage predicted correctly:')
print('Training:', train_score[1] * 100, '%')
print('Testing:', test_score[1] * 100, '%')