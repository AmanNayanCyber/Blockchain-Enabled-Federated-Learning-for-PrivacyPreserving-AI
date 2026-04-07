import torch


def add_gaussian_noise(state_dict, noise_scale=0.01):
    noisy_state_dict = {}

    for key, value in state_dict.items():
        if torch.is_tensor(value):
            noise = torch.randn_like(value) * noise_scale
            noisy_state_dict[key] = value + noise
        else:
            noisy_state_dict[key] = value

    return noisy_state_dict


def clip_state_dict(state_dict, max_norm=1.0):
    total_norm = 0.0
    for value in state_dict.values():
        if torch.is_tensor(value):
            total_norm += value.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        clipped_state_dict = {}
        for key, value in state_dict.items():
            if torch.is_tensor(value):
                clipped_state_dict[key] = value * scale
            else:
                clipped_state_dict[key] = value
        return clipped_state_dict

    return state_dict
