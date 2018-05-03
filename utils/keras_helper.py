import matplotlib.pyplot as plt
import random
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def create_activation_plot():
    # y = 0.5x + 1.5
    x = list(range(-10,10))
    logits = [x*0.5+1.5 for x in range(-10,10)]
    activation = [sigmoid(y) for y in logits ]
    f = plt.figure(figsize=(10,3))
    ax = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    ax.plot(x,logits)
    ax.set_title('linear function 0.5*x+1.5')
    ax2.plot(x,activation)
    ax2.set_title("sigmoid(0.5x+1.5)")
    plt.savefig('img/sigmoid_activation.png')

def plot_accuracy(history):
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
# summarize history for loss
def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

def plot_history(history):
    f = plt.figure(figsize=(15,3))
    ax = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    ax.plot(history.history['acc'])
    ax.set_title('model accuracy')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    ax2.plot(history.history['loss'])
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    plt.show()
    
def create_subplot(plot,data,label):
    rnd = random.randint(0,60000)
    if isinstance(label[rnd],np.ndarray):
        label = np.argmax(label[rnd])
    else:
        label = label[rnd]
    plot.imshow(data[rnd], cmap=plt.get_cmap('gray'))
    plot.set_title('Label =  '+str(label))
    
def show_random_mnist(train_data, train_label):
    f = plt.figure(figsize=(15,5))
    ax = f.add_subplot(131)
    ax2 = f.add_subplot(132)
    ax3 = f.add_subplot(133)
    create_subplot(ax,train_data,train_label)
    create_subplot(ax2,train_data,train_label)
    create_subplot(ax3,train_data,train_label)
    plt.show()

def create_exploded_mnist(flatten = False, one_hot = True, normalize = True):
    #Load the MNIST Dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    #X_train_img = X_train.reshape((-1,28,28))
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    X_train2 = np.zeros((n_train, 80,80))
    for i in range(n_train):
        col_shift = 40 + random.randint(-39,11)
        row_shift = 40 + random.randint(-39,11)
        X_train2[i,row_shift:row_shift+28,col_shift:col_shift+28] = X_train[i]
    
    X_test2 = np.zeros((n_test, 80,80))
    for i in range(n_test):
        col_shift = 40 + random.randint(-39,11)
        row_shift = 40 + random.randint(-39,11)
        X_test2[i,row_shift:row_shift+28,col_shift:col_shift+28] = X_test[i]

    X_train = X_train2
    X_test = X_test2
    
    if normalize:
        # normalize inputs from 0-255 to 0-1
        X_train = X_train / 255
        X_test = X_test / 255
        
    if one_hot:
        # one hot encode outputs
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
    
    if flatten:
        # flatten 28*28 images to a 784 vector for each image
        X_train = X_train.reshape(X_train.shape[0], -1).astype('float32')
        X_test = X_test.reshape(X_test.shape[0], -1).astype('float32')
    
    return (X_train, y_train, X_test, y_test) 

