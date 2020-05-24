from sklearn.datasets import load_iris

import torch.nn as nn
import torch.optim as optim
import torch as t
import torch.nn.functional as f

import numpy as np

from matplotlib import pyplot as plt


if __name__ == "__main__":
    data_set = load_iris()

    data = data_set["data"]
    targets = data_set["target"]

    network = nn.Sequential(
        nn.Linear(4, 20),
        nn.Sigmoid(),
        nn.Linear(20, 3),
        nn.Sigmoid()
    )

    optimizer = optim.Adam(network.parameters())

    accuracy = []
    for epoch in range(2000):
        # chose 20 random indexes of data to be used in test
        training_set_indexes = np.random.choice(len(data), 130, replace=False)
        # absorb test indexes to get training indexes
        test_set_indexes = np.delete(np.arange(len(data)), training_set_indexes)

        test_set = data[test_set_indexes]
        training_set = data[training_set_indexes]

        test_set_targets = targets[test_set_indexes]
        training_set_targets = targets[training_set_indexes]

        training_data = t.tensor(training_set).float()

        training_set_targets_ = np.zeros([len(training_set_targets), 3])
        for index, output in enumerate(training_set_targets):
            training_set_targets_[index, output] = 1
        expected_output = t.tensor(training_set_targets_).float()

        optimizer.zero_grad()

        output = network.forward(training_data)

        loss = f.mse_loss(output, expected_output)
        loss.backward()

        optimizer.step()

        correct_predictions = 0
        for test_data, test_target in zip(test_set, test_set_targets):
            test_data = t.tensor(test_data).float()

            optimizer.zero_grad()

            output = network(test_data)

            biggest_element = -1
            biggest_element_index = -1
            for index, element in enumerate(output):
                if biggest_element < element:
                    biggest_element = element
                    biggest_element_index = index

            label = biggest_element_index

            if label == test_target:
                correct_predictions += 1

        print("epoch " + str(epoch) + " accuracy: " + str(correct_predictions / 20))
        accuracy.append(correct_predictions / 20)

    batch = 50
    accuracy_ = [sum(accuracy[n:n+batch]) / batch for n in range(0, len(accuracy), batch)]
    plt.plot(list(range(len(accuracy_))), accuracy_)
    plt.show()
