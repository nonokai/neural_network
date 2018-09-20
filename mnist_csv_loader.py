import numpy as np
import json

def load(filename1, filename2):

    training_data = []
    validation_data = []

    with open(filename1) as fileobj:
        count = 0
        while True:
            line = fileobj.readline().rstrip()
            if line:
                line = list(map(int, line.split(",")))
                x = np.array(line[1:], ndmin = 2).T / 255.0
                y = np.zeros((10, 1))
                y[line[0]] = 1
                count += 1
                if count <= 50000:
                    training_data.append((x, y))
                else:
                    validation_data.append((x, y))
            else:
                print("load completed : training_data and validation_data")
                break

    test_data = []

    with open(filename2) as fileobj:
        while True:
            line = fileobj.readline().rstrip()
            if line:
                line = list(map(int, line.split(",")))
                x = np.array(line[1:], ndmin = 2).T / 255.0
                y = np.zeros((10, 1))
                y[line[0]] = 1
                test_data.append((x, y))
            else:
                print("load completed : test_data")
                break


    return training_data, validation_data, test_data

