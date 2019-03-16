import torch
import torch.nn.functional as F
from torch import nn


class Struc2Vec(nn.Module):

    def __init__(self, p=64):
        super(Struc2Vec, self).__init__()
        self.theta1 = torch.ones((1, p), requires_grad=True)
        self.theta2 = torch.ones((p, p), requires_grad=True)
        self.theta3 = torch.ones((p, p), requires_grad=True)
        self.theta4 = torch.ones((1, 1, p), requires_grad=True)

    def forward(self, x: torch.Tensor, mu: torch.Tensor, w: torch.Tensor, adj_mat: torch.Tensor):
        t1 = self.theta1.mul(x)  # (N, p)

        t2 = adj_mat.mm(mu)
        t2 = t2.mm(self.theta2)  # (N, p)

        t3 = self.theta4.mul(w.unsqueeze(2))
        t3 = F.relu(t3)
        t3 = t3.sum(dim=1)
        t3 = t3.mm(self.theta3)  # (N, p)

        return F.relu(torch.add(t1, t2.add(t3)))
