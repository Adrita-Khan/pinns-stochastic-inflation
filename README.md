# Physics-Informed Neural Networks for Stochastic Inflation


<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Issues](https://img.shields.io/badge/issues-open-brightgreen)](https://github.com/Adrita-Khan/pinns-stochastic-inflation/issues)
[![GitHub stars](https://img.shields.io/github/stars/Adrita-Khan/pinns-stochastic-inflation)](https://github.com/Adrita-Khan/pinns-stochastic-inflation/stargazers)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=26&duration=3000&pause=800&color=ff8c00&center=true&vCenter=true&repeat=true&width=820&lines=Physics-Informed+Neural+Networks;Stochastic+Inflation+Dynamics;Quantum+Fluctuations+in+Cosmic+Inflation;Langevin+Equation+Modeling;PINNs+for+Cosmology;Neural+PDE+Solvers;Computational+Cosmology;Machine+Learning+for+Early+Universe" alt="Typing SVG" />
</p>

<p align="center">
<i>
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=800&color=39FF14&center=true&vCenter=true&repeat=true&width=900&lines=Exploratory+Machine+Learning+for+Inflationary+Cosmology;Stochastic+Processes+in+the+Early+Universe;Physics-Constrained+Deep+Learning;Quantum+Noise+Driven+Inflation;Alternative+to+Monte+Carlo+Simulations;Scalable+Neural+Solvers+for+Cosmology" alt="Typing SVG" />
</i>
</p>

> **Note:** The feasibility of this project is currently being explored, and it remains subject to continuous advancements and modifications.



## Overview
This project explores the use of **Physics-Informed Neural Networks (PINNs)** to model the dynamics of stochastic inflation, providing an efficient alternative to traditional simulations.

## Background

### The Problem
During cosmic inflation, quantum fluctuations of the inflaton field lead to stochastic dynamics described by the Langevin equation:

```
dφ/dt = -V'(φ)/(3H) + (H^(3/2)/(2π)) ξ(t)
```

where:
- φ is the inflaton field
- V(φ) is the inflaton potential
- H is the Hubble parameter during inflation
- ξ(t) is Gaussian white noise with
  ⟨ξ(t)⟩ = 0 and ⟨ξ(t) ξ(t')⟩ = δ(t − t')
  
## Traditional Approach: Monte Carlo Simulations
- **Process**: Generate billions of stochastic realizations
- **Challenges**:
  - Computationally expensive (hours of computation)
  - Memory intensive (gigabytes of data)
  - Statistical convergence requires many realizations
  - Not reusable across different parameter sets

## PINN Approach
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

| Author(s) | Year | Title | Source | Links |
|---------|------|-------|--------|-------|
| Raissi, Perdikaris, Karniadakis | 2019 | Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs | *Journal of Computational Physics* 378, 686–707 | [PDF](https://indico.ictp.it/event/a12197/session/4/contribution/2/material/0/0.pdf) · [DOI](https://doi.org/10.1016/j.jcp.2018.10.045) |
| Starobinsky | 1986 | Stochastic de Sitter (inflationary) stage in the early universe | *Lecture Notes in Physics* 246, 107–126 | [DOI](https://doi.org/10.1007/3-540-16452-9_6) · [Background](https://en.wikipedia.org/wiki/Starobinsky_inflation) |
| Linde | 2008 | Inflationary cosmology | *Lecture Notes in Physics* 738, 1–54 | [DOI](https://doi.org/10.1007/978-3-540-74353-8_1) · [Book](https://www.taylorfrancis.com/books/mono/10.1201/9780367807788/particle-physics-inflationary-cosmology-andrei-linde) |
| Baumann | 2009 | TASI lectures on inflation | arXiv:0907.5424 | [arXiv](https://arxiv.org/abs/0907.5424) · [Notes](https://faculty.sites.iastate.edu/hliu/files/inline-files/PINN_RPK_2019_1.pdf) |
| Mishra | 2024 | Cosmic inflation: Background dynamics, quantum fluctuations and reheating | arXiv:2403.10606 | [arXiv](https://arxiv.org/abs/2403.10606) · [INSPIRE](https://inspirehep.net/files/21cef7e2d2ef50ab5ee95967bf608911) |
| Senatore | 2013 | Lectures on inflation | CERN & Stanford Summer School | [Slides](https://ivanik3.narod.ru/Kosmology/LindeParticle.pdf) · [Indico](https://indico.ictp.it/event/a12197/session/4/contribution/2/material/0/0.pdf) |


| # | Source | Link |
|---|--------|------|
| 1 | SSRN | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5083429 |
| 2 | arXiv:2106.09728 | https://arxiv.org/pdf/2106.09728 |
| 3 | Nature | https://www.nature.com/articles/s41540-025-00500-6 |
| 4 | EmergentMind | https://www.emergentmind.com/topics/physics-informed-stochastic-perturbation-scheme |
| 5 | arXiv:2109.01621 | https://arxiv.org/pdf/2109.01621 |
| 6 | ICCS 2025 | https://www.iccs-meeting.org/archive/iccs2025/papers/159090262.pdf |
| 7 | arXiv:2204.03859 | https://arxiv.org/pdf/2204.03859 |
| 8 | arXiv:2010.12685 | https://arxiv.org/pdf/2010.12685 |


## Physics-Informed Neural Networks (PINNs) in Cosmology


| # | Title | Category | Paper Link | GitHub | Dataset | Year |
|---|-------|----------|------------|--------|---------|------|
| 1 | Cosmology-informed Neural Networks to infer dark energy equation-of-state | Dark Energy | [arXiv:2508.12032](https://arxiv.org/abs/2508.12032) | ❌ | [Pantheon+](https://archive.stsci.edu/prepds/ps1cosmo/) | 2025 |
| 2 | Inferring Cosmological Parameters with Evidential Physics-Informed Neural Networks | Dark Energy | [arXiv:2509.24327](https://arxiv.org/abs/2509.24327) | ❌ | [Pantheon+](https://archive.stsci.edu/prepds/ps1cosmo/) | 2025 |
| 3 | Unraveling particle dark matter with Physics-Informed Neural Networks | Dark Matter | [arXiv:2502.17597](https://arxiv.org/abs/2502.17597) | ❌ | ❌ | 2025 |
| 4 | SPINN: Advancing Cosmological Simulations of Fuzzy Dark Matter | Dark Matter | [arXiv:2506.02957](https://arxiv.org/abs/2506.02957) | ❌ | ❌ | 2025 |
| 5 | Physics-informed neural networks in recreation of hydrodynamic simulations | Dark Matter | [MNRAS 527(2)](https://academic.oup.com/mnras/article/527/2/3381/7342487) | ❌ | [SIMBA](https://simba.roe.ac.uk) | 2023 |
| 6 | LensPINN: Learning Dark Matter Morphology in Lensing | Dark Matter | [NeurIPS 2024](https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_78.pdf) | [✅](https://github.com/ML4SCI/DeepLense) | ❌ | 2024 |
| 7 | Physics-Informed Neural Networks for Galactic Gravitational Potentials | Galactic Dynamics | [NeurIPS 2025](https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_120.pdf) | ✅ | ❌ | 2025 |
| 8 | Cosmology-informed neural networks for background dynamics of Universe | Background Dynamics | [PRD 107](https://link.aps.org/doi/10.1103/PhysRevD.107.063523) | [✅](https://github.com/at-chantada/cosmo-nets) | ❌ | 2023 |
| 9 | Template-Fitting Meets Deep Learning: Redshift Estimation | Redshift | [arXiv:2507.00866](https://arxiv.org/abs/2507.00866) | ❌ | PREML | 2025 |

## Datasets

| Dataset | Link | Used By |
|---------|------|---------|
| Pantheon+ Type Ia Supernovae | [STScI Archive](https://archive.stsci.edu/prepds/ps1cosmo/) | Papers #1, #2 |
| SIMBA Cosmological Simulations | [SIMBA Portal](https://simba.roe.ac.uk) | Paper #5 |
| PREML (Hyper Suprime-Cam PDR3) | Not publicly linked | Paper #9 |

## General Resources

| Repository | Link |
|------------|------|
| maziarraissi/PINNs | [GitHub](https://github.com/maziarraissi/PINNs) |
| omniscientoctopus/Physics-Informed-Neural-Networks | [GitHub](https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks) |
| georgestein/ml-in-cosmology | [GitHub](https://github.com/georgestein/ml-in-cosmology) |



