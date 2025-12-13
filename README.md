# Gossip Learning under Different Network Topologies

This repository contains the code and experiments for our research project on **gossip-based decentralized learning under different network topologies**.  
The goal of this project is to study how **network structure** and **mixing strategies** affect convergence speed, stability, and final accuracy in fully decentralized machine learning systems.

We focus on gossip learning as an alternative to centralized federated learning, where nodes communicate only with their neighbors and no central server exists.

---

## Project Overview

In this project, we:
- Implement a **fully decentralized gossip learning framework** from scratch
- Train models on **non-IID Fashion-MNIST data**
- Compare performance across three network topologies:
  - **Erdős–Rényi (ER)** random graphs
  - **Watts–Strogatz (WS)** small-world networks
  - **Barabási–Albert (BA)** scale-free networks
- Evaluate two mixing strategies:
  - **Uniform averaging**
  - **Metropolis–Hastings (MH) weighting**
- Analyze convergence speed, final accuracy, and communication behavior

The project is research-oriented and designed to support reproducibility and further experimentation.

---

## Repository Structure

.
├── src/
│   ├── config.py           # configuration dictionary defining experiment parameters
│   ├── run_experiments.py  # main entry point to run the experiment suite
│   ├── generate_plots.py   # script to create visualizations from saved results
│   ├── experiments.py      # function to run gossip experiments and save metrics
│   ├── training.py         # core gossip training loop
│   ├── network.py          # GossipNetwork class implementing mixing and evaluation
│   ├── node.py             # Node class handling local training and evaluation
│   ├── data.py             # data partitioning utilities (Dirichlet split)
│   ├── models.py           # SimpleMLP model definition
│   └── ...                 # other helpers (if added)
└── results/                # directory where plots are saved (created after running)
└── experiment_data/        # directory where CSV results are saved (created after running)

