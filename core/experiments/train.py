import torch
import torch.optim as optim
import torch.nn as nn

from core.model import SANA
from core.controller import MetaController

model = SANA(10, 32, 1)
controller = MetaController()

optimizer = optim.Adam(model.parameters(), lr=controller.lr)
loss_fn = nn.MSELoss()

for epoch in range(100):
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)

    optimizer.zero_grad()

    y_pred = model(x, controller.stress)
    loss = loss_fn(y_pred, y)

    loss.backward()
    optimizer.step()

    stress, lr = controller.adjust(loss.item())
    for g in optimizer.param_groups:
        g['lr'] = lr

    print(f"{epoch}: loss={loss.item():.4f}, stress={stress:.4f}, lr={lr:.6f}")
