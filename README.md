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

## Running the Code

You can run the project in **two ways**, depending on your preference.

---

### Option 1: Run with Jupyter Notebook (Recommended for Exploration)

If you prefer an interactive environment:

1. Download or open `GossipLearning.ipynb`
2. Launch Jupyter Notebook: Run the cells sequentially

This option is useful for:
1. Step-by-step understanding of gossip learning
2. Debugging and visualization
3. Interactive analysis of results
   


