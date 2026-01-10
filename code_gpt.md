Below is a **complete, self-contained PINN-style implementation** tailored to **stochastic inflation**, with careful comments explaining:

* the **physics** (Langevin/Fokker–Planck formulation),
* how **stochasticity is handled inside a PINN**,
* how to **train once** and then generate **billions of realizations cheaply**,
* how to do **training + inference**.

I will keep it realistic, scalable, and *actually useful for cosmology*, not a toy ODE.

---

# 1. Physics background (what we are solving)

### Stochastic inflation (slow-roll, single field)

The coarse-grained inflaton field (\phi(N)) (as a function of e-folds (N)) obeys the **Langevin equation**:

[
\frac{d\phi}{dN}
================

* \frac{V'(\phi)}{3 H^2(\phi)}

- \frac{H(\phi)}{2\pi},\xi(N)
  ]

where

* (H^2(\phi) = \frac{V(\phi)}{3 M_{\rm Pl}^2}),
* (\xi(N)) is Gaussian white noise:
  [
  \langle \xi(N)\xi(N') \rangle = \delta(N-N')
  ]

---

### Equivalent deterministic equation (key PINN idea)

Instead of simulating **billions of noisy trajectories**, we solve the **Fokker–Planck equation** for the **probability density** (P(\phi, N)):

[
\boxed{
\frac{\partial P}{\partial N}
=============================

\frac{\partial}{\partial \phi}
\left[
\frac{V'(\phi)}{3 H^2(\phi)} P
\right]
+
\frac{1}{2}
\frac{\partial^2}{\partial \phi^2}
\left[
\left(\frac{H(\phi)}{2\pi}\right)^2 P
\right]
}
]

This is:

* **deterministic**
* **low dimensional**
* ideal for **PINNs**

Once (P(\phi,N)) is learned → sampling is **trivial and fast**.

---

# 2. Strategy

1. PINN learns (P_\theta(\phi, N))
2. Physics loss enforces the Fokker–Planck PDE
3. Initial condition enforced at (N=0)
4. After training:

   * generate **billions of samples** by fast sampling from (P(\phi,N))
   * no SDE integration needed

---

# 3. Complete PINN implementation (PyTorch)

## 3.1 Imports

```python
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
```

---

## 3.2 Inflation potential (example: quadratic)

You can replace this with **Starobinsky, hilltop, axion**, etc.

```python
# Planck mass (set to 1 for natural units)
Mpl = 1.0

def V(phi):
    """Inflationary potential V(phi)"""
    m = 1e-6
    return 0.5 * m**2 * phi**2

def dV_dphi(phi):
    m = 1e-6
    return m**2 * phi

def H(phi):
    return torch.sqrt(V(phi) / (3.0 * Mpl**2))
```

---

## 3.3 PINN architecture

We model **log P** for numerical stability:
[
P(\phi,N) = e^{u_\theta(\phi,N)}
]

```python
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

        self.activation = nn.Tanh()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)  # outputs log P
```

---

## 3.4 Physics loss: Fokker–Planck equation

```python
def fokker_planck_residual(model, phi, N):
    """
    Computes PDE residual of the Fokker–Planck equation
    """
    phi.requires_grad_(True)
    N.requires_grad_(True)

    inputs = torch.cat([phi, N], dim=1)
    logP = model(inputs)
    P = torch.exp(logP)

    # First derivatives
    P_N = autograd.grad(P, N, torch.ones_like(P), create_graph=True)[0]
    P_phi = autograd.grad(P, phi, torch.ones_like(P), create_graph=True)[0]

    # Second derivative
    P_phi2 = autograd.grad(P_phi, phi, torch.ones_like(P_phi), create_graph=True)[0]

    drift = dV_dphi(phi) / (3.0 * H(phi)**2)
    diffusion = (H(phi) / (2.0 * np.pi))**2

    rhs = (
        autograd.grad(drift * P, phi, torch.ones_like(P), create_graph=True)[0]
        + 0.5 * diffusion * P_phi2
    )

    residual = P_N - rhs
    return residual
```

---

## 3.5 Initial condition

At (N=0), start sharply peaked at (\phi_0):

[
P(\phi,0) = \mathcal{N}(\phi_0, \sigma^2)
]

```python
def initial_condition_loss(model, phi0=15.0, sigma=0.1, n_points=200):
    phi = torch.linspace(phi0-1, phi0+1, n_points).unsqueeze(1)
    N = torch.zeros_like(phi)

    logP_pred = model(torch.cat([phi, N], dim=1))
    P_pred = torch.exp(logP_pred)

    P_true = torch.exp(-0.5*((phi-phi0)/sigma)**2) / (sigma*np.sqrt(2*np.pi))
    return torch.mean((P_pred - P_true)**2)
```

---

## 3.6 Training loop

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

model = PINN([2, 64, 64, 64, 1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5000):

    # Collocation points
    phi = torch.rand(1000, 1) * 20.0
    N = torch.rand(1000, 1) * 60.0

    phi, N = phi.to(device), N.to(device)

    residual = fokker_planck_residual(model, phi, N)
    physics_loss = torch.mean(residual**2)

    ic_loss = initial_condition_loss(model)

    loss = physics_loss + ic_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss = {loss.item():.3e}")
```

---

# 4. Fast inference: generating **billions** of realizations

Once trained, you **never integrate the SDE again**.

---

## 4.1 Sampling from learned PDF

```python
@torch.no_grad()
def sample_phi(model, N_eval, n_samples=10_000_000):
    """
    Sample phi values at fixed N from learned PDF
    """
    phi_grid = torch.linspace(0, 20, 5000).unsqueeze(1)
    N_grid = torch.ones_like(phi_grid) * N_eval

    logP = model(torch.cat([phi_grid, N_grid], dim=1))
    P = torch.exp(logP).cpu().numpy().flatten()

    P /= np.trapz(P, phi_grid.numpy().flatten())

    samples = np.random.choice(
        phi_grid.numpy().flatten(),
        size=n_samples,
        p=P / P.sum()
    )
    return samples
```

---

## 4.2 Example usage

```python
samples = sample_phi(model, N_eval=50.0, n_samples=1_000_000_000)

print("Mean phi:", samples.mean())
print("Variance:", samples.var())
```

This step is:

* **embarrassingly parallel**
* GPU-independent
* can scale to **billions** trivially

---

# 5. Why this works (important)

| Traditional             | PINN                       |
| ----------------------- | -------------------------- |
| (10^9) SDE integrations | 1 PDE solve                |
| Noise every step        | Noise encoded in diffusion |
| Hours–days              | Minutes                    |
| Trajectory based        | Distribution based         |

---

# 6. Extensions you can publish

You are *very well positioned* to turn this into a paper if you:

1. Add:

   * multi-field inflation
   * eternal inflation boundary
2. Compare:

   * PINN vs brute-force Monte Carlo
3. Use:

   * conditional PINNs for different potentials
4. Accelerate with:

   * Fourier features
   * JAX + vmap

---

If you want, I can next:

* rewrite this **in JAX** (much faster),
* add **eternal inflation absorbing boundaries**,
* or help you **turn this into a GitHub repo + paper outline**.
