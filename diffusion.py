import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange

from roberta import RoBERTaForMaskedLM, get_batch, mask_token_id, device

# diffusion hyperparameters
N_STEPS = 10  # number of noise levels (including 0 and 1)
LEARNING_RATE = 3e-4


def mask_batch_dynamic(x, mask_prob):
    """Like get_batch but for arbitrary mask_prob.

    returns (x_masked, labels) where labels=-100 for unmasked.
    """
    labels = x.clone()
    prob = torch.rand(x.shape, device=device)
    mask = prob < mask_prob
    labels[~mask] = -100
    x2 = x.clone()
    x2[mask] = mask_token_id
    return x2, labels


class MaskedLanguageDiffusion(nn.Module):
    def __init__(self, n_steps=N_STEPS):
        super().__init__()
        self.n_steps = n_steps
        # linearly spaced mask probabilities from 0.0 → 1.0
        self.mask_probs = torch.linspace(0.0, 1.0, n_steps).tolist()

        # one RoBERTa model per noise level
        self.models = nn.ModuleList(
            [RoBERTaForMaskedLM().to(device) for _ in range(n_steps)]
        )
        # independent optimizers
        self.optims = [
            torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE) for m in self.models
        ]

    def forward_diffusion(self, x0):
        """Given a clean batch x0, produce the full noising trajectory.

        [x0, x1, x2, …, x_{T-1}], where x_t has mask_prob = mask_probs[t].
        """
        xs = [x0]
        for p in self.mask_probs[1:]:
            x_noisy, _ = mask_batch_dynamic(xs[-1], p)
            xs.append(x_noisy)
        return xs

    def train_step(self, x0):
        """Forward-noise, For each t, train model[t] to predict x_{t-1} from x_t.

        Returns average loss over all levels.
        """
        xs = self.forward_diffusion(x0)
        losses = []
        for t in range(1, self.n_steps):
            xt = xs[t]
            x_prev = xs[t - 1]
            _, loss = self.models[t](xt, labels=x_prev)
            self.optims[t].zero_grad()
            loss.backward()
            self.optims[t].step()
            losses.append(loss.item())
        return sum(losses) / len(losses)

    @torch.no_grad()
    def sample(self, x_T):
        """Given a masked batch, iteratively apply each model[t] to recover x_{t-1}.

        returns [x0_pred, x1_pred, …, x_T]
        """
        xs = [x_T]
        for t in reversed(range(1, self.n_steps)):
            xt = xs[-1]
            logits, _ = self.models[t](xt)  # no labels here
            probs = F.softmax(logits, dim=-1)
            x_pred = torch.argmax(probs, dim=-1)
            xs.append(x_pred)
        return list(reversed(xs))


if __name__ == "__main__":
    diffusion = MaskedLanguageDiffusion(n_steps=N_STEPS)
    pbar = trange(1000, desc="Diffusion Training", unit="iter")
    for it in pbar:
        xb, _ = get_batch("train")  # x0 batch
        loss = diffusion.train_step(xb)
        # update every 100 iters
        if it % 100 == 0:
            pbar.set_postfix(diffusion_loss=f"{loss:.4f}")

    # Save entire diffusion state
    torch.save(
        {
            "mask_probs": diffusion.mask_probs,
            "model_states": [m.state_dict() for m in diffusion.models],
        },
        "diffusion_weights.pt",
    )
    print("Diffusion weights saved to diffusion_weights.pt")
