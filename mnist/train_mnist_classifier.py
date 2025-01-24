from neural_network.model import Model, Network, Layer
from neural_network.functions import *
import pandas as pd
import numpy as np

def print_img(bytes):
    for i, byte in zip(range(1, 785, 1), bytes):
            if byte < 50.0 / 255.0:
                print(".", end='')
            elif byte < 100.0 / 255.0:
                print("o", end='')
            else:
                print("@", end='')
            if i % 28 == 0:
                print("\n", end='')

def read_idx_image(file) -> np.ndarray:
    rows = []
    with open(file, 'rb') as f:
        header = f.read(16)
        
        while (image := f.read(784)):
            rows.append([float(byte) for byte in image])
    
    return np.array(rows, dtype=float)

def read_idx_label(file) -> np.ndarray:
    rows = []
    with open(file, 'rb') as f:
        header = f.read(8)
        
        while (label := f.read(1)):
            rows.append([int.from_bytes(label, byteorder='big')])
    
    return np.array(rows, dtype=np.uint8)



def main():
    path = './mnist/'

    try:
        model = Model(model_type='multi-classifier')
        model.net.add_layer(Layer("input", 784, activation=None, weight_initialization=None))
        model.net.add_layer(Layer("hidden", 64, activation=ReLU, weight_initialization=he_initialization))
        model.net.add_layer(Layer("hidden", 64, activation=ReLU, weight_initialization=he_initialization))
        model.net.add_layer(Layer("output", 10, activation=softmax, weight_initialization=he_initialization))

    except Exception as e:
        print(e.message)
        exit()
    
    train_images = read_idx_image(path + 'data/train-images.idx3-ubyte') / 255
    train_labels = read_idx_label(path + 'data/train-labels.idx1-ubyte')
    test_images = read_idx_image(path + 'data/t10k-images.idx3-ubyte') / 255
    test_labels = read_idx_label(path + 'data/t10k-labels.idx1-ubyte')

    print("TRAIN DATA:")
    for i in range(3):
        print(train_labels[i])
        print_img(train_images[i])
        print()
    print("----------------\n")

    print("TEST DATA:")
    for i in range(3):
        print(test_labels[i])
        print_img(test_images[i])
        print()
    print("----------------\n")
    
    training_data = []
    test_data = []
    for X, Y in zip(train_images, train_labels):
        training_data.append((X.reshape((784, 1)), Y[0]))
    for X, Y in zip(test_images, test_labels):
        test_data.append((X.reshape((784, 1)), Y[0]))
    

    #Training data needs to be a list of tupels where each tuple is (column_vec of features, int of label).
    # Example. Image of a nine: ([210, 255, 123, ...], 9)
    # The label is used for hot encodig internally.
    model.fit(training_data, validation_data=test_data, epochs=10, eta=0.001, batch_size=10)
    model.plot_training()
    model.save_model(path + "mnist.pkl")

if __name__ == '__main__':
    main()

   
        
         
            
