from detoxify import Detoxify
import torch
import torch.nn.functional as F


def safety_score(text):
    scores = Detoxify("original").predict(text)
    return scores["toxicity"]  # Return toxicity score


def gumbel_softmax(logits, tau=1.0):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    return F.softmax((logits + gumbel_noise) / tau, dim=-1)