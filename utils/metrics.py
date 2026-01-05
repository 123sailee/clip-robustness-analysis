import torch
import torch.nn.functional as F
from typing import Tuple

def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)

def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    topk = logits.topk(k, dim=1).indices
    correct = (topk == labels.unsqueeze(1)).any(dim=1).sum().item()
    return correct / labels.size(0)

def mean_cosine_similarity(image_features: torch.Tensor, text_features: torch.Tensor) -> float:
    """Assumes L2-normalized features or will normalize internally."""
    image_features = F.normalize(image_features, dim=1)
    text_features = F.normalize(text_features, dim=1)
    # compute pairwise cosine similarity for corresponding rows
    sims = (image_features * text_features).sum(dim=1)
    return sims.mean().item()
