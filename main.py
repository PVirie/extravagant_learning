import tensorflow as tf
import numpy as np
import os

root = os.path.dirname(os.path.abspath(__file__))

mnist = tf.keras.datasets.fashion_mnist
# mnist.load_data(os.path.join(root, "artifacts", "fashion_mnist.npz"))
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

x_train = train_images.reshape(-1, 28, 28).astype(np.float32) / 255.0
x_test = test_images.reshape(-1, 28, 28).astype(np.float32) / 255.0
# normalize data
mean = np.mean(x_train, axis=0, keepdims=True)
std = np.std(x_train, axis=0, keepdims=True)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std


y_train = train_labels
y_test = test_labels
print(x_train.shape, x_test.shape)


if __name__ == "__main__":
    print("main")
