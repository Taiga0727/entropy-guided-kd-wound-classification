"""Entropy-guided KD loss (EGAS), extracted VERBATIM from notebooks/entropy_guided_kd.ipynb (cell 56).

Not edited and not separately unit-tested. Differences from the paper's text are listed in the README
("Implementation notes"): the KL target is teacher_probs / T (not softmax(teacher_logits / T)), the
"contrastive" term is a cosine embedding loss on the logits (not on features), and the entropy
normalisation is a per-batch min-max.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# The notebook creates 'device' once (cuda if available) and the loss reads it as a global.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EntropyWeightedKDLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=2.0):
        super(EntropyWeightedKDLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        # Compute probabilities from logits
        teacher_probs = torch.softmax(teacher_logits, dim=1)

        # Compute entropy for each sample
        entropy = -torch.sum(teacher_probs * torch.log(teacher_probs + 1e-9), dim=1)

        # Normalize entropy
        entropy_min = entropy.min()
        entropy_max = entropy.max()
        entropy_scaled = (entropy - entropy_min) / (entropy_max - entropy_min)

        # Dynamic Temperature
        T = torch.clamp(1.0 + self.beta * entropy_scaled, min=1.0, max=5.0)

        # Adaptive Alpha (ให้ KL สำคัญเมื่อ entropy ต่ำ และ CE สำคัญเมื่อ entropy สูง)
        adaptive_alpha = torch.sigmoid(2 - entropy)

        # Compute KL Loss with dynamic T
        kl_loss = self.criterion_kl(
            torch.log_softmax(student_logits / T.unsqueeze(1), dim=1),
            teacher_probs / T.unsqueeze(1)
        )

        # Compute Cross-Entropy Loss
        ce_loss = self.criterion_ce(student_logits, labels)

        # Compute Contrastive Loss (Fix)
        contrastive_loss = F.cosine_embedding_loss(
            student_logits, teacher_logits, torch.ones_like(labels, dtype=torch.float).to(device)
        )

        # Combine losses (Fix: Reduce to scalar)
        total_loss = (adaptive_alpha * kl_loss * T.mean().item() ** 2) + ((1 - adaptive_alpha) * ce_loss) + (0.1 * contrastive_loss)

        return total_loss.mean()  # Ensure scalar output
