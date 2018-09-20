import network2
import numpy as np
from mnist_csv_loader import load

training_data, validation_data, test_data = load("mnist_train.csv", "mnist_test.csv")

net = network2.Network([784, 30, 10])
evaluation_cost, evaluation_accuracy, training_cost, training_accuracy \
= net.SGD(  training_data, 30, 10, 0.5,
            lmbda=5.0,
            evaluation_data=validation_data,
            monitor_evaluation_cost=True,
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True)
