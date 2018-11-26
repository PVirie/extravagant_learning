import numpy as np


class Single_mode_Semantic_memory:

    def __init__(self, pool_size, input_depth, output_depth):
        print("init")
        self.pool_size = pool_size
        self.input_depth = input_depth
        self.output_depth = output_depth

        self.allocation = np.zeros([pool_size])
        self.v = np.zeros([input_depth, pool_size])
        self.h = np.zeros([pool_size, output_depth])

    def learn(self, data, label):
        print("learn")
        self.stem(data)

    def stem(self, data):
        print("stem")
        q = np.matmul(data, self.v)
        p = np.amax(q, axis=1)
        selection = np.argmax(q, axis=1)[0]
        return self.h[selection, :] * p


if __name__ == '__main__':
    print("symantic.py")
