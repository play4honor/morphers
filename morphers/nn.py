import torch
from torch import nn


# We need this horrible thing so Morphers can use it to return the right shape.
class Unsqueezer(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)

# Time2Vec Module
class Time2Vec(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.size = size

        self.weight = nn.Parameter(torch.zeros([1, self.size]))
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros([1, self.size]))
        nn.init.kaiming_uniform_(self.bias, nonlinearity="relu")

    def forward(self, x):
        linear_term = self.weight[:, 0] * x + self.bias[:, 0]
        periodic_terms = torch.sin(self.weight[:, 1:] * x + self.bias[:, 1:])

        return torch.cat([linear_term, periodic_terms], dim=-1)

# Contrastive Predictive Coding loss for BigIntegerizer
class CPCLoss(torch.nn.Module):
    def __init__(self, embedding_layer, n_negative_samples):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.n_negative_samples = n_negative_samples

    def forward(self, input, target):
        embeddings = self.embedding_layer.weight

        # n x s x 1 x es
        target_embeddings = embeddings[target.unsqueeze(-1)]

        # n x s x (k-1)
        negative_indices = torch.randint(
            low=0,
            high=embeddings.shape[0] - 1,
            size=[input.shape[0], input.shape[1], self.n_negative_samples],
            device=input.device,
        )

        negative_indices += (negative_indices >= target.unsqueeze(-1)).long()
        # n x s x (k-1) x es
        negative_embeddings = embeddings[negative_indices]

        # n x s x k x es
        all_embeddings = torch.cat([target_embeddings, negative_embeddings], dim=-2)
        all_embeddings = nn.functional.normalize(all_embeddings)

        # input is n x s x es
        normalized_input = nn.functional.normalize(input).unsqueeze(dim=-2)
        # n x s x k
        logits = (all_embeddings * normalized_input).sum(dim=-1)

        return nn.functional.cross_entropy(
            logits.transpose(-1, -2),  # n x k x s
            torch.zeros(
                [logits.shape[0], logits.shape[1]],
                dtype=torch.int64,
                device=logits.device,
            ),  # n x s
            reduction="none",
        )


# Rank Scale Layer
# (Leave me alone, the paper doesn't name it.)
class RankScaleTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = torch.nn.Parameter(torch.tensor([0.5]))
        self.src_scale = torch.nn.Parameter(torch.ones([1]))
        self.src_shift = torch.nn.Parameter(torch.zeros([1]))
        self.rank_scale = torch.nn.Parameter(torch.ones([1]))
        self.rank_shift = torch.nn.Parameter(torch.zeros([1]))

    def forward(self, x):
        # TKTK Is this the right way to clip weight?
        # x is n x 2
        self.weight = torch.nn.Parameter(torch.clamp(self.weight, min=0.0, max=1.0))
        x = self.weight * ((x[:, 0] + self.src_shift) * self.src_scale) + (
            1 - self.weight
        ) * ((x[:, 1] + self.rank_shift) * self.rank_scale)
        return x
