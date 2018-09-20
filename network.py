import numpy as np
import random

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Network:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # 入力層以外のbiasの縦ベクトルの配列
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # x:出力ノード数
        # y:入力ノード数
        # の重みの配列（y行x列）
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        # a:入力(numpy配列で行数が入力要素数の縦ベクトル)
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        # 返り値はニューラルネットワークの出力(numpy配列の縦ベクトル)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        # training_data:(x, y)のタプルのリスト、xは入力データ、yは教師信号
        # epochs:エポック数(訓練データを何巡するか)
        # mini_batch_size:訓練入力いくつで勾配を計算するか
        # eta:学習率
        # test_data:training_dataと同様
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            # training_dataをshuffleしてmini_batch_sizeに分割する
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:(k + mini_batch_size)]
                for k in range(0, n, mini_batch_size)
            ]
            # mini_batchそれぞれに対して勾配を計算し、weightとbiasを更新する
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            # テストデータがある場合には正答率を表示
            if test_data:
                print(f"Epoch {j+1}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j+1} complete")

    def update_mini_batch(self, mini_batch, eta):
        # mini_batch:(x,y)のタプル
        # eta:学習率
        # まず0で埋められたbiasとweightと同じ形の配列を準備
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        # まず0で埋められたbiasとweightと同じ形の配列を準備
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 活性化関数に入れた後の値を保管する箱を準備
        # 初期値はx
        activation = x
        activations = [x]

        # 活性化関数に入れる前の値を保管する箱を準備
        zs = []

        # feedforward(値を記録しながら)
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # cost_derivative(コスト関数の勾配、すなわち出力層の誤差)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-(l-1)].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-(l+1)].T)

        return(nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for x, y in test_data]
        return sum([int(x == y) for x, y in test_results])

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)
