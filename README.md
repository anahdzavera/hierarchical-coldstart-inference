# Hierarchical Bayesian Cold-Start Forecasting
### Cross-domain inference under informative censorship

> *"The same generative problem appears in an e-commerce warehouse in Mexico City,  
> a clinical trial in Basel, and a telescope array in the Atacama Desert."*

---

## Overview

This repository develops a **Bayesian hierarchical model for demand forecasting of new products** with no individual sales history — the *cold-start problem* — and demonstrates that the same generative structure governs three apparently unrelated domains:

| Domain | Object | Observable | Latent quantity | Censorship mechanism |
|---|---|---|---|---|
| E-commerce | New SKU | Weekly units sold | Latent demand | Stock constraint: `y = min(d, stock)` |
| Pharmacology | New drug | Trial response data | True efficacy | Dropout / loss to follow-up |
| Astronomy | New transient | Flux from light curve | Intrinsic luminosity | Magnitude limit of instrument |

The central claim of this project is that all three are instances of the same inference problem: **cold-start inference under informative censorship with hierarchical information transfer**. The methodology developed here applies to all three.

---

## Motivation

### The business problem

Avera is a Mexican e-commerce company selling home appliances across Amazon, Mercado Libre, Walmart, Liverpool, Elektra, and Coppel. A post-mortem analysis of 49 new SKUs launched since April 2024 identified 21 out-of-stock events and approximately 19.4M MXN in estimated lost sales.

The root cause: purchasing decisions were made without formal demand forecasts, using only conservative rules of thumb. New products ran out of stock within 8–9 weeks, before reorder cycles could respond.

### The scientific problem

Forecasting demand for a new product is structurally identical to a class of problems studied in statistics, pharmacology, and astrophysics: **inference about a latent process when observations are censored by the capacity of the measurement instrument**, and the only available prior information comes from a population of similar objects.

This equivalence is not merely analogical — the generative models share the same probabilistic graph.

---

## The Generative Model

### Level 0 — Indices and data structure

```
k ∈ {1, ..., K}       SKUs (or drugs, or transients)
c(k) ∈ {1, ..., C}    Category of object k (product family, drug class, transient type)
t ∈ {1, ..., T_k}     Time steps since launch / trial start / detection
```

### Level 1 — Latent demand (the quantity we want to infer)

True demand is never directly observed. It is modeled as:

```
λ_{k,t} = μ_k · r(t; α_k, β_k) · s(t)
```

Where:

- **`μ_k`** — steady-state demand level for object `k` (units/week in e-commerce; response rate in pharma; intrinsic flux in astronomy)
- **`r(t; α_k, β_k)`** — ramp-up function: a monotone increasing curve from 0 to 1 modeling the transition from launch to steady state. Parameterized as the Weibull CDF:

```
r(t) = 1 − exp(−(t / α_k)^β_k)
```

  - `α_k` (scale): controls *when* the object reaches steady state
  - `β_k` (shape): controls *how* the transition happens
    - `β < 1`: fast early growth, gradual saturation (viral products, fast-rising transients)
    - `β = 1`: exponential approach to steady state
    - `β > 1`: slow start, then acceleration (products requiring discovery; slow-rising light curves)

- **`s(t)`** — deterministic seasonal multiplier (promotional events, calendar effects)

### Level 2 — Hierarchical prior (information transfer from the population)

Rather than estimating `μ_k` from scratch for each new object, objects in the same category share a distribution. This is the mechanism by which information from *similar* objects informs predictions about a *new* object.

```
# SKU level
μ_k     ~ LogNormal(μ_{c(k)}, σ_{c(k)})
α_k     ~ LogNormal(ᾱ_{c(k)}, σ_α)
β_k     ~ LogNormal(β̄_{c(k)}, σ_β)

# Category level
μ_c     ~ LogNormal(μ_0, σ_0)

# Global hyperpriors
μ_0     ~ Normal(...)
σ_0     ~ HalfNormal(...)
```

LogNormal is used throughout because demand, scale, and shape parameters are strictly positive and naturally right-skewed.

**Astronomical parallel:** This is structurally identical to the BayeSN model for Type Ia supernovae (Mandel et al. 2022), where each SN Ia's light curve parameters are drawn from a population distribution inferred from the full SN Ia catalog. A newly detected supernova with only 3–4 flux measurements gets its posterior from the population prior — exactly as a new SKU with 2 weeks of sales history borrows from its category.

### Level 3 — Censorship likelihood (the critical structural feature)

Observed sales are not latent demand. They are censored by available stock:

```
y_{k,t} = min(d_{k,t}, stock_{k,t})
```

Where `d_{k,t}` is realized demand, drawn from a Negative Binomial distribution to accommodate overdispersion (correlated purchase behavior, marketing bursts):

```
d_{k,t} ~ NegativeBinomial(λ_{k,t}, φ)
```

The likelihood has two regimes:

```
                ⎧ P(d_{k,t} = y_{k,t})           if stock_{k,t} > y_{k,t}   [complete observation]
L_{k,t}  =     ⎨
                ⎩ P(d_{k,t} ≥ stock_{k,t})        if stock_{k,t} = y_{k,t}   [right-censored]
```

A week where the product ran out of stock is not a week of zero demand — it is a week where demand was *at least* as large as the stock on hand. Treating it as zero demand (as naive models do) systematically underestimates the latent demand level.

**This is formally identical to right-censored survival analysis**, where a subject who leaves the study is not assumed to have had the event at exit — only that they survived until that point.

**Astronomical parallel:** A source fainter than the instrument's limiting magnitude is not absent — it is undetected. The likelihood contribution is `P(flux > detection limit)`, the survival function evaluated at the threshold. Same mathematical object.

---

## Generative Graph Summary

```
μ_0, σ_0         ← global hyperpriors
      │
      ▼
μ_c, σ_c         ← category level (product family / drug class / transient type)
      │
      ▼
μ_k, α_k, β_k   ← object level (SKU / compound / transient)
      │
      ▼
λ_{k,t} = μ_k · r(t; α_k, β_k) · s(t)    ← latent demand rate
      │
      ▼
d_{k,t} ~ NegBin(λ_{k,t}, φ)              ← realized demand
      │
      ▼
y_{k,t} = min(d_{k,t}, stock_{k,t})       ← censored observation
```

---

## Cross-Domain Equivalence Table

| Model component | E-commerce (Avera) | Pharmacology | Astronomy |
|---|---|---|---|
| `μ_k` | Steady-state weekly demand | Baseline response rate | Intrinsic luminosity |
| `r(t)` | Sales ramp-up post-launch | Dose-response curve | Light curve rise phase |
| `α_k` | Weeks to reach regime | Time to half-maximal effect | Rise timescale |
| `β_k` | Shape of adoption curve | Hill coefficient | Light curve shape index |
| `s(t)` | Buen Fin / Hot Sale multiplier | Dosing schedule | Observation cadence |
| `stock_{k,t}` | Available inventory | Sample size / trial capacity | Instrument limiting magnitude |
| OOS week | Stockout | Patient dropout | Detection below threshold |
| Category prior | Product family history | Drug class meta-analysis | SN Ia population model |
| `φ` (dispersion) | Purchase clustering | Biological variability | Intrinsic scatter |

---

## Repository Structure

```
.
├── README.md
├── data/
│   ├── raw/                    # Original sales and inventory data (not committed)
│   └── processed/              # Cleaned weekly grids per SKU
├── src/
│   ├── 04_modelo_semanal.py    # Current frequentist model (baseline)
│   ├── model/
│   │   ├── generative.py       # PyMC model definition
│   │   ├── priors.py           # Prior specifications by category
│   │   └── censorship.py       # Custom censored likelihood
│   └── analysis/
│       ├── diagnostics.py      # ArviZ posterior diagnostics
│       └── cross_domain.py     # Equivalence demonstrations
├── notebooks/
│   ├── 01_eda_skus.ipynb       # Exploratory analysis of 49 SKUs
│   ├── 02_model_fitting.ipynb  # PyMC sampling and convergence
│   └── 03_cross_domain.ipynb   # Pharmacology and astronomy parallels
├── paper/
│   ├── main.tex                # LaTeX manuscript
│   └── figures/                # Generated plots
└── requirements.txt
```

---

## Implementation Stack

- **Python 3.11** (pyenv)
- **PyMC 5.x** — probabilistic programming, NUTS sampling
- **ArviZ** — posterior diagnostics, visualization
- **NumPy / Pandas** — data manipulation
- **Matplotlib / Seaborn** — figures

---

## Status

| Phase | Description | Status |
|---|---|---|
| 0 | Post-mortem frequentist model (49 SKUs, OOS detection) | ✅ Complete |
| 1 | Generative model specification | 🔄 In progress |
| 2 | PyMC implementation with real Avera data | ⬜ Pending |
| 3 | Cross-domain equivalence formalization | ⬜ Pending |
| 4 | Academic paper draft | ⬜ Pending |

---

## References (preliminary)

- Mandel et al. (2022). *BayeSN: Hierarchical Bayesian SED Model for Type Ia Supernovae.* MNRAS.
- Ibrahim & Chen (2000). *Power Prior Distributions for Regression Models.* Statistical Science.
- Gelman et al. (2013). *Bayesian Data Analysis*, 3rd ed. CRC Press.
- Salvatier, Wiecki & Fonnesbeck (2016). *Probabilistic programming in Python using PyMC3.* PeerJ Computer Science.
- Bass (1969). *A New Product Growth Model for Consumer Durables.* Management Science.

---

## Authors

Ana Paula — BI Analyst, Avera | Applied Mathematics

*Project developed in collaboration with Alan Domínguez (Avera) and Héctor (academic direction).*
