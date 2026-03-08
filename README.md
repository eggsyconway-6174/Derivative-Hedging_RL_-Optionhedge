# 🚀 Advanced Derivative Hedging with Deep Reinforcement Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production‑ready implementation of an **RL‑based option hedging agent** that learns to dynamically adjust hedge ratios, minimize transaction costs, and control drawdowns. The project compares **PPO** against traditional **delta** and **gamma hedging** strategies using real market data (with robust fallback to synthetic data).

---

## 📌 Overview

Traditional hedging approaches like delta or gamma hedging assume static market parameters and fail to adapt to changing conditions. This project applies **Reinforcement Learning** to develop an adaptive hedging strategy that continuously learns from market dynamics.

The agent is trained in a custom OpenAI‑gym‑style environment that simulates realistic market behaviour, including:
- Stochastic volatility
- Transaction costs & market impact
- Greeks (delta, gamma, theta, vega) calculated via Black‑Scholes
- Option‑specific features (moneyness, time to expiry)

After training, the RL agent is evaluated against baseline delta and gamma hedging strategies using key financial metrics.

---

## ✨ Key Features

- **Multi‑source data pipeline**: Automatically fetches data from Stooq, pandas‑datareader, Alpha Vantage, and yfinance (with exponential backoff). Falls back to realistic synthetic data if all sources fail.
- **Rich state space**: Includes price, volatility, Greeks, time to maturity, moneyness, current position, and PnL.
- **Continuous action space**: Hedge ratio adjustments in `[-1, 1]`.
- **Sophisticated reward function**: Penalises drawdowns, transaction costs, and hedging error; includes asymmetric loss penalty.
- **PPO agent** with distributional critic and GAE.
- **Benchmark strategies**: Classic delta hedging and gamma‑aware hedging.
- **Comprehensive evaluation**: Sharpe ratio, max drawdown, hedging error, transaction costs, portfolio variance.
- **Visualisation**: Training curves and strategy comparison plots.
- **Production‑ready code**: Type hints, logging, error handling, GPU support.

---

## ⚙️ How It Works

### State Space (10 dimensions)
| Index | Feature                  | Normalisation         |
|-------|--------------------------|-----------------------|
| 0     | Normalised price         | `price / initial_price - 1` |
| 1     | Normalised volatility    | `vol / 0.5`           |
| 2     | Delta                    | raw                   |
| 3     | Gamma (×100)             | `gamma * 100`         |
| 4     | Theta                    | raw                   |
| 5     | Vega (÷100)              | `vega / 100`          |
| 6     | Normalised time to expiry| `T * 252 / 30`        |
| 7     | Moneyness                | `price / strike - 1`  |
| 8     | Normalised position      | `position / 1000`     |
| 9     | Normalised PnL           | `pnl / 10000`         |

### Action Space
Continuous value in `[-1, 1]` representing the target delta (hedge ratio). The agent learns to output the desired delta, and the environment adjusts the position accordingly.

### Reward Function
reward = pnl_change - transaction_cost

risk_aversion * drawdown_penalty

risk_aversion * hedging_error * 100

risk_aversion * abs(gamma) * 1000
if pnl_change < 0: reward *= 2 (asymmetric loss penalty)


### Training
- Algorithm: PPO (Proximal Policy Optimization) with 5 epochs per update, GAE(λ=0.95), clipped surrogate objective.
- Episodes: 150 (each up to 100 steps).
- Optimiser: Adam with learning rate 3e-4.

### Baselines
- **Delta hedging**: Maintains delta‑neutral position by setting hedge ratio = -current delta.
- **Gamma hedging**: Reduces position when gamma exposure is high.

---

