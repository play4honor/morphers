import torch


# We need this horrible thing so Morphers can use it to return the right shape.
class Unsqueezer(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


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
