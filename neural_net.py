import torch.nn as nn


def neural_net():
    model = nn.Sequential()
    model.add_module("Layer1", nn.Linear(5, 500))
    model.add_module("Activation1", nn.ReLU())
    model.add_module("Layer2", nn.Linear(500, 500))
    model.add_module("Activation2", nn.ReLU())
    model.add_module("Output", nn.Linear(500, 3))

    return model
