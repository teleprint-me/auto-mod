import torch
from ..model import GPT


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        # Compute your custom loss function here
        return torch.F.mse_loss(logits.flatten(), targets.flatten())


def train(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    criterion: CustomLoss,
    dataloader: str,
    device: torch.device,
):
    model.train()

    for batch in dataloader:
        input_ids = batch[0].to(device)
        targets = batch[1].to(device)

        optimizer.zero_grad()

        output = model(input_ids)

        loss = criterion(output, targets)

        loss.backward()

        optimizer.step()
