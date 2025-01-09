import torch


def noise_augmentation(data, noise_factor=0.2):
    """
    Add random noise to input data.

    :param data: Input data (e.g., images).
    :param noise_factor: Factor to scale the added noise.
    :return: Noisy data.
    """
    noisy_data = data + noise_factor * torch.randn_like(data)
    noisy_data = torch.clamp(noisy_data, 0.0, 1.0)  # Ensure data remains in valid range
    return noisy_data


def add_label_noise(labels, num_classes, noise_ratio=0.1):
    """
    Add noise to labels by randomly flipping some labels.

    :param labels: Original labels.
    :param num_classes: Number of classes.
    :param noise_ratio: Proportion of labels to randomize.
    :return: Labels with noise added.
    """
    noisy_labels = labels.clone()
    num_noisy = int(noise_ratio * len(labels))
    noisy_indices = torch.randperm(len(labels))[:num_noisy]
    noisy_labels[noisy_indices] = torch.randint(0, num_classes, (len(noisy_indices),))
    return noisy_labels
