import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np 
import sklearn.covariance


def get_msp_scores(model, images):
    logits = model(images)
    probs = F.softmax(logits, dim=1)
    msp = probs.max(dim=1).values

    scores = - msp  # The larger MSP, the smaller uncertainty

    return logits, scores

def get_rep_norm_scores(model, images):
    reps = model.forward_features(images)
    logits = model.forward_classifier(reps)
    rep_norm = reps.norm(dim=1)

    scores = - rep_norm
    return logits, scores

def get_energy_scores(model, images):
    logits = model(images)
    probs = F.softmax(logits, dim=1)
    energy = torch.logsumexp(probs, dim=1)

    scores = - energy
    return logits, scores
