import torch
import torch.nn.functional as F


def pg_loss(sample_probs, sample_reward):
    """vanilla policy gradient loss"""
    return - (sample_reward * sample_probs.mean())


def pgb_loss(sample_probs, sample_reward, baseline):
    """policy gradient with baseline loss"""
    return (baseline - sample_reward) * sample_probs.mean()


def nanmean(x):
    return x[~torch.isnan(x)].mean()


def entropy(p):
    entropy = -p * torch.log2(p)
    out = nanmean(entropy)
    return out


def compute_loss(args, probs, sample_probs, s_reward, a_reward):
    if args.loss == "pg":
        loss = pg_loss(sample_probs, s_reward)
    elif args.loss == "pgb":
        loss = pgb_loss(sample_probs, s_reward, a_reward)
    else:
        raise ValueError(f"unknown loss function: {args.loss}")
    return loss
