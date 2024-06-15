import torch


def choose_options(p, temperature: float = 1.0):
    """Function used for generation
    Takes an array of logits, not probabilities, sorry."""
    assert temperature >= 0.0
    # p is n x k
    if temperature == 0.0:
        choices = torch.argmax(p, dim=-1)
    else:
        p = torch.softmax(p / temperature, dim=-1)
        agg_p = p.cumsum(dim=1)
        rand = torch.rand(agg_p.shape[0], 1).to(agg_p)
        p_arrays = torch.cat([agg_p, rand], dim=1)
        # n x 1
        ranks = torch.argsort(torch.argsort(p_arrays, dim=-1), dim=-1)
        choices = ranks[:, -1]
    return choices
