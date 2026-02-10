import torch
import torch.nn.functional as F


def logistic_loss(z, labels):
    """Calculates the logistic loss."""
    return F.softplus(z * -labels)


def sigmoid_loss(z, labels):
    """Calculates the sigmoid loss."""
    return torch.sigmoid(z * -labels)


# A dictionary to easily select the hardness function from config
hardness_functions = {
    "logistic": logistic_loss,
    "sigmoid": sigmoid_loss,
}
