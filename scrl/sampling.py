import torch
import random
from collections import defaultdict
from torch.distributions import Categorical
from scrl.model import labels_to_summary
from nltk import word_tokenize


def sample_from_policy(
        input_ids,
        probs,
        eps_greedy=0,
        force_diff=True,
        diff_trials=1000,
    ):
    m = Categorical(probs)
    argmax_labels = torch.argmax(probs, dim=2)
    sample_labels = m.sample()

    if eps_greedy > 0:
        sample_labels = apply_epsilon_greedy(probs, sample_labels, eps_greedy).to("cuda")
    if force_diff:
        for _ in range(diff_trials):
            if (argmax_labels == sample_labels).all():
                sample_labels = m.sample()
            else:
                break

    sample_probs = m.log_prob(sample_labels)
    return sample_probs, sample_labels


def apply_epsilon_greedy(probs, sample_labels, eps=0.1):
    batch_size = probs.size(0)
    new_sample_labels = []
    for i in range(batch_size):
        s_labels = sample_labels[i].tolist()
        new_labels = []
        for l in s_labels:
            if random.random() <= eps:
                l = random.randint(0, 1)
            new_labels.append(l)
        new_sample_labels.append(new_labels)
    sample_labels = torch.tensor(new_sample_labels)
    return sample_labels


def best_of_k_samples(
        manager,
        tokenizer,
        reward_generator,
        input_ids,
        batch,
        probs,
        k_samples=50,
        return_all=False
    ):
    batch_size = probs.size(0)

    prob_batches = []
    summary_batches = []
    reward_batches = []
    detail_batches = []
    label_batches = []
    for _ in range(k_samples):
        sample_probs, sample_labels = sample_from_policy(
            input_ids,
            probs,
        )
        sample_summaries = labels_to_summary(
            input_ids, sample_labels, tokenizer
        )
        sample_rewards, sample_details = reward_generator(
            batch["document"], sample_summaries
        )

        prob_batches.append(sample_probs)
        summary_batches.append(sample_summaries)
        reward_batches.append(sample_rewards)
        detail_batches.append(sample_details)
        label_batches.append(sample_labels)


    best_indices = []
    for i in range(batch_size):
        rewards = [reward_batches[j][i] for j in range(k_samples)]
        scored = sorted(enumerate(rewards), key=lambda x: x[1], reverse=True)
        best_idx = scored[0][0]
        best_indices.append(best_idx)

    sample_probs = torch.stack([prob_batches[j][i] for i, j in enumerate(best_indices)])
    sample_summaries = [summary_batches[j][i] for i, j in enumerate(best_indices)]
    sample_rewards = [reward_batches[j][i] for i, j in enumerate(best_indices)]
    sample_labels = torch.stack([label_batches[j][i] for i, j in enumerate(best_indices)])

    sample_details = []
    for i, j in enumerate(best_indices):
        detail_keys = sorted(detail_batches[0].keys())
        details = defaultdict(list)
        for k in detail_keys:
            details[k].append(detail_batches[j][k][i])
        sample_details.append(details)

    sample_data = {
        "probs": prob_batches,
        "rewards": reward_batches,
        "summaries": summary_batches,
        "details": detail_batches,
        "labels": label_batches,
    }
    return sample_probs, sample_summaries, sample_rewards, sample_details, sample_labels, sample_data
