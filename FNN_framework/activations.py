import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableELU(nn.Module):
    """
    Learnable parameters:
        - alpha : controls negative region sharpness
        - beta  : scales the entire output
        - gamma : scales input before exponent (controls curvature)
    """

    def __init__(self,
                 init_alpha=1.0,
                 init_beta=1.0,
                 init_gamma=1.0,
                 learnable=True):
        super().__init__()

        if learnable:
            self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
            self.beta  = nn.Parameter(torch.tensor(init_beta,  dtype=torch.float32))
            self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))
        else:
            # buffer = constant, not trainable
            self.register_buffer("alpha", torch.tensor(init_alpha))
            self.register_buffer("beta",  torch.tensor(init_beta))
            self.register_buffer("gamma", torch.tensor(init_gamma))

    def forward(self, x):
        '''
        Formula:
        L-ELU(x) = beta * (x if x > 0 else alpha * (exp(gamma * x) - 1))
        '''
        # positive region: x
        pos = torch.relu(x)

        # negative region: alpha * (exp(gamma*x) - 1)
        neg = -torch.relu(-x)
        neg = self.alpha * (torch.exp(self.gamma * neg) - 1.0)

        # combine + scale entire output by beta
        return self.beta * (pos + neg)


class ParamHardSigmoid(nn.Module):
    """
    Parametric Hard-Sigmoid for bounded regression.
    
    Formula:
        y = clamp(k * x + b, L, U)

    Where:
        - k (slope) and b (shift) are learnable
        - L and U are fixed output bounds (e.g., 0 to 100)
        
    Why this works:
        - k learns intensity sensitivity
        - b learns midpoint (what counts as “neutral emotion”)
        - avoids floor collapse (0 vs 1-3 issue)
        - avoids ceiling flattening (85-100 saturated region)
        - preserves smoothness while filtering noisy labels
    """

    def __init__(self, L=0.0, U=100.0, init_k=0.2, init_b=50.0):
        super().__init__()
        
        # learnable slope and bias
        self.k = nn.Parameter(torch.tensor(init_k, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(init_b, dtype=torch.float32))
        
        # output bounds (fixed)
        self.L = L
        self.U = U

    def forward(self, x):
        # linear transform
        y = self.k * x + self.b
        
        # clamp to physical hormone range
        y = torch.clamp(y, self.L, self.U)
        return y
