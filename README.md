# Physics-Informed Neural Networks for Stochastic Inflation

## Overview
This project explores the use of **Physics-Informed Neural Networks (PINNs)** to model the dynamics of stochastic inflation, providing an efficient alternative to traditional Monte Carlo simulations.

## Background

### The Problem
During cosmic inflation, quantum fluctuations of the inflaton field lead to stochastic dynamics described by the Langevin equation:

```
dφ/dt = -V'(φ)/(3H) + (H/2π)ξ(t)
```

where:
- φ is the inflaton field
- V(φ) is the inflaton potential
- H is the Hubble parameter during inflation
- ξ(t) is Gaussian white noise

### Traditional Approach: Monte Carlo Simulations
- **Process**: Generate billions of stochastic realizations
- **Challenges**:
  - Computationally expensive (hours of computation)
  - Memory intensive (gigabytes of data)
  - Statistical convergence requires many realizations
  - Not reusable across different parameter sets

### PINN Approach
Physics-Informed Neural Networks embed physical laws directly into the neural network training process through the loss function:

```
Loss = Loss_IC + Loss_physics + Loss_data
```

where:
- `Loss_IC`: Enforces initial conditions
- `Loss_physics`: Enforces the governing differential equation
- `Loss_data`: Matches available data (if any)

## Methodology

### PINN Loss Function
The total loss combines three components:

1. **Initial Condition Loss**:
   ```
   L_IC = ||φ_pred(0) - φ₀||²
   ```

2. **Physics Loss** (enforces the Langevin equation):
   ```
   L_physics = ||∂φ/∂t + V'(φ)/(3H)||²
   ```

3. **Data Loss** (if reference data available):
   ```
   L_data = ||φ_pred - φ_ref||²
   ```

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).  
   *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.*  
   Journal of Computational Physics, **378**, 686–707.

2. Starobinsky, A. A. (1986).  
   *Stochastic de Sitter (inflationary) stage in the early universe.*  
   In *Current Topics in Field Theory, Quantum Gravity and Strings* (pp. 107–126). Springer.

3. Linde, A. (2008).  
   *Inflationary cosmology.*  
   Physics of Particles and Nuclei, **39**(7), 971–1015.

4. Baumann, D. (2009).  
   *TASI lectures on inflation.*  
   arXiv:0907.5424  
   https://arxiv.org/abs/0907.5424

5. Mishra, S. S. (2024).  
   *Cosmic inflation: Background dynamics, quantum fluctuations and reheating.*  
   arXiv:2403.10606  
   https://arxiv.org/abs/2403.10606

