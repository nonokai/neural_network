import network
import numpy as np

training_data = []

with open("./mnist_train.csv") as fileobj:
    while True:
        line = fileobj.readline().rstrip()
        if line:
            line = list(map(int, line.split(",")))
            x = np.array(line[1:], ndmin = 2).T / 255
            y = np.zeros((10, 1))
            y[line[0]] = 1
            training_data.append((x, y))
        else:
            print("訓練用データを読み込み完了しました。")
            break

test_data = []

with open("./mnist_test.csv") as fileobj:
    while True:
        line = fileobj.readline().rstrip()
        if line:
            line = list(map(int, line.split(",")))
            x = np.array(line[1:], ndmin = 2).T / 255
            y = np.zeros((10, 1))
            y[line[0]] = 1
            test_data.append((x, y))
        else:
            print("テストデータを読み込み完了しました。")
            break

net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 100.0, test_data=test_data)


