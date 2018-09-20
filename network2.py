import numpy as np
import random

def sigmoid(x):
    return 1.0 / (1.0+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))

class CrossEntropyCost:

    @staticmethod
    def fn(a, y):
        return np.nan_to_num(np.sum(-y*np.log(a) - (1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return (a-y)

class QuadraticCost:

    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        return (a-y) * sigmoid_prime(z)

class Network:

    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        # 入力層以外のbiasの縦ベクトルの配列
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        # x:出力ノード数
        # y:入力ノード数
        # の重みの配列（y行x列）
        # 平均0、標準偏差1/√ninのガウス分布で初期化
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        # a:入力(numpy配列で行数が入力要素数の縦ベクトル)
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        # 返り値はニューラルネットワークの出力(numpy配列の縦ベクトル)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        # training_data:(x, y)のタプルのリスト、xは入力データ、yは教師信号
        # epochs:エポック数(訓練データを何巡するか)
        # mini_batch_size:訓練入力いくつで勾配を計算するか
        # eta:学習率
        # lmbda:正規化パラメーター
        # evaluation_data:training_dataと同様
        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            # training_dataをshuffleしてmini_batch_sizeに分割する
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:(k+mini_batch_size)]
                for k in range(0, n, mini_batch_size)
            ]
            # mini_batchそれぞれに対して勾配を計算し、weightとbiasを更新する
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))

            print(f"Epoch {j+1} training complete")

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print(f"Cost on training data: {cost}")

            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data)
                training_accuracy.append(accuracy)
                print(f"Accuracy on training data: {accuracy} / {n}")

            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda)
                evaluation_cost.append(cost)
                print(f"Cost on evaluation data: {cost}")

            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print(f"Accuracy on evaluation data: {accuracy} / {n_data}")

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        # mini_batch:(x,y)のタプル
        # eta:学習率
        # lmbda:正規化パラメーター
        # n:訓練データ数
        # まず0で埋められたbiasとweightと同じ形の配列を準備
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w - (eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        # まず0で埋められたbiasとweightと同じ形の配列を準備
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 活性化関数に入れた後の値を保管する変数、リストを準備
        # 初期値はx
        activation = x
        activations = [x]

        # 活性化関数に入れる前の値を保管するリストを準備
        zs = []

        # feedforward(値を記録しながら)
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # cost_derivative(コスト関数の勾配、すなわち出力層の誤差)
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-(l-1)].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-(l+1)].T)

        return(nabla_b, nabla_w)

    def accuracy(self, data):
        results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]

        return sum([int(x == y) for (x, y) in results])

    def total_cost(self, data, lmbda):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost
