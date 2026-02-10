import torch


def entropy_loss(p: torch.Tensor) -> torch.Tensor:
    """Computes the entropy loss for predicted probabilities.

    This loss encourages the model to make high-confidence predictions on
    unlabeled data by minimizing the entropy of the predictions.
    A prediction with high entropy is one where p is close to 0.5.
    A prediction with low entropy is one where p is close to 0 or 1.

    The entropy is calculated as: - (p * log(p) + (1-p) * log(1-p)).
    The loss is the mean of the entropy over the batch.

    Args:
        p (torch.Tensor): A tensor of predicted probabilities, which should
                          be the output of a sigmoid function. Values must
                          be in the range [0, 1].

    Returns:
        torch.Tensor: The mean entropy loss.
    """
    p = torch.clamp(p, min=1e-7, max=1 - 1e-7)
    return -torch.mean(p * torch.log(p) + (1 - p) * torch.log(1 - p))
