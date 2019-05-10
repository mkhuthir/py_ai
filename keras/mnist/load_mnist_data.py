#!/usr/bin/python3

# MNIST (python-mnist) opens the ubyte files that store the datasets
from mnist import MNIST

# array converts native python array-like objects e.g lists into numpy arrays
from numpy import array

# tqdm displays progress bars for looping procedures.
from tqdm import tqdm

def get_mnist_data():
    """
    Open mnist files and load the data in form of lists.
    Combine the training and testing into one.
    """
    data = MNIST('./data')
    train_images, train_labels = data.load_training()
    test_images, test_labels = data.load_testing()
    images = train_images + test_images
    labels = list(train_labels) + list(test_labels)
    return images, labels

def one_hot_encode(integer, array_length):
    """
    One-Hot-Encode a digit.
    For example, (integer=0,array_length=10) reterns base=[1,0,0,0,0,0,0,0,0,0]
    """
    base = [0 for i in range(array_length)]
    base[integer] = 1
    return base


def normalize_and_reshape(image):
    """
    Reshaping converts the one-dimensional array into a
    two-dimensional array just like an image.
    
    Normalisation is done by dividing all pixel values
    by 255 to make values between 0 and 1.
    """
    return array([float(i) / 255 for i in image]).reshape(28, 28)


def split_data():
    """
    Split data into training and testing.
    Training will have 40000 datapoints.
    Testing will have 30000 datapoints.
    Reshape the arrays as the network would expect.
    """
    x_train, y_train = images[:40000].reshape(
        40000, 28, 28, 1), labels[:40000]
    x_test, y_test = images[40000:].reshape(
        30000, 28, 28, 1), labels[40000:]
    return x_test, x_train, y_test, y_train


# read mnist data
print("\nLoading mnist data from files...")
images, labels = get_mnist_data()

print('One-Hot-Encode Outputs')
labels = array([one_hot_encode(label, 10) for label in tqdm(labels)])

print('Normalize and reshape input images')
images = array([normalize_and_reshape(image) for image in tqdm(images)])

print('Spliting 70K data points into 40K for training and 30K for testing')
x_test, x_train, y_test, y_train = split_data()

print('Data Loading completed.\n')