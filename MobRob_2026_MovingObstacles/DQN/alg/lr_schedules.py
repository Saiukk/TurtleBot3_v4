"""Learning-rate helpers shared by SB3 wrappers."""

import math


def cosine_episode_learning_rate(initial_lr, min_lr, completed_episodes, total_episodes):
    if total_episodes <= 1:
        return min_lr

    progress = min(max(completed_episodes / float(total_episodes - 1), 0.0), 1.0)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (initial_lr - min_lr) * cosine_factor


def set_optimizer_learning_rate(optimizer, learning_rate):
    if optimizer is None:
        return

    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def set_model_learning_rate(model, learning_rate):
    """Update the optimizers used by PPO/SAC to the same learning rate."""

    optimizers = []

    policy = getattr(model, "policy", None)
    if policy is not None:
        optimizers.append(getattr(policy, "optimizer", None))

    actor = getattr(model, "actor", None)
    if actor is not None:
        optimizers.append(getattr(actor, "optimizer", None))

    critic = getattr(model, "critic", None)
    if critic is not None:
        optimizers.append(getattr(critic, "optimizer", None))

    ent_coef_optimizer = getattr(model, "ent_coef_optimizer", None)
    if ent_coef_optimizer is not None:
        optimizers.append(ent_coef_optimizer)

    for optimizer in optimizers:
        set_optimizer_learning_rate(optimizer, learning_rate)