import network3 as network
import numpy as np
import matplotlib.pyplot as plt
from mnist_csv_loader import load
import cost_function as cf
import activation_function as af

training_data, validation_data, test_data = load("mnist_train.csv", "mnist_test.csv")

net = network.Network([784, 30, 10], cost=cf.CrossEntropyCost, hidden_af=af.Sigmoid, output_af=af.Sigmoid)

evaluation_cost, evaluation_accuracy, training_cost, training_accuracy \
= net.SGD(  training_data, 30, 10, 0.05,
            lmbda=5.0,
            evaluation_data=validation_data,
            monitor_evaluation_cost=True,
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True)

fig = plt.figure()

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.set_title("cost")
ax1.plot(training_cost, label="training")
ax1.plot(evaluation_cost, label="test")
ax1.legend()

ax2.set_title("accuracy")
ax2.plot(training_accuracy, label="training")
ax2.plot(evaluation_accuracy, label="test")
ax2.legend()

fig.tight_layout()
fig.savefig("./png/output.png")
fig.show()

plt.close(fig)
