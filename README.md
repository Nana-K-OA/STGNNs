# Traffic Forecasting on METR-LA using Spatial-Temporal Graph Neural Networks

Comparison of three spatial-temporal graph neural networks for traffic speed
forecasting on the METR-LA dataset (207 sensors, Los Angeles), implemented with
the [Torch Spatiotemporal (TSL)](https://github.com/TorchSpatiotemporal/tsl) library.

## Overview

Traffic is shaped by temporal trends and by spatial dependencies across the road
network. This project compares:

| Model | Role |
|-------|------|
| **TimeThenSpace (TTS)** | Simple baseline (GRU for time, then diffusion convolution for space) |
| **GraphWaveNet (GWN)** | Learns a self-adaptive adjacency matrix with dilated temporal convolutions |
| **AGCRN** | Adaptive Graph Convolutional Recurrent Network with node-specific parameters |

Objectives:
1. Evaluate each model at 15, 30 and 60-minute forecasting horizons.
2. Visualise the spatial dependencies learned by GWN and AGCRN.
3. Compare my GWN results with the original GraphWaveNet paper (Wu et al., 2019).

## Dataset and setup

- **Dataset:** METR-LA, traffic speed from 207 sensors (downloaded automatically by TSL)
- **Split:** 70% train / 10% validation / 20% test (temporal split)
- **Normalisation:** StandardScaler
- **Window / horizon:** 12 input steps, 12 forecast steps (5-minute intervals)
- **Metrics:** MAE, MAPE, MSE (masked, to handle missing values)
- **Optimiser:** Adam, lr = 0.001, batch size 64

## Results

Average test performance across all 207 sensors.

| Metric | Horizon | TTS | GWN | AGCRN |
|--------|---------|-----|-----|-------|
| MAE | 15 min | 2.96 | 2.76 | 2.85 |
| | 30 min | 3.55 | 3.16 | 3.23 |
| | 60 min | 4.42 | 3.68 | 3.67 |
| MSE | 15 min | 33.10 | 27.25 | 30.51 |
| | 30 min | 50.73 | 39.16 | 42.81 |
| | 60 min | 78.73 | 55.08 | 58.08 |

**Training:** TTS ~8.5 min (50 epochs), GWN ~66 min (30 epochs), AGCRN ~36.7 min (40 epochs).

### Comparison with the original GraphWaveNet paper

| Metric | Horizon | Mine | Paper |
|--------|---------|------|-------|
| MAE | 15 / 30 / 60 min | 2.76 / 3.16 / 3.68 | 2.69 / 3.07 / 3.53 |
| MSE* | 15 / 30 / 60 min | 27.25 / 39.16 / 55.08 | 26.52 / 38.69 / 54.31 |

*Paper RMSE values were squared to compare with MSE.

### Key findings

- GWN was the most accurate overall, especially at longer horizons.
- AGCRN trained in roughly half the time of GWN with slightly higher error.
- GWN's learned adjacency captures non-local dependencies; TTS relies on a fixed,
  distance-based graph.
- My GWN results are within a small margin of the original paper.

See the full write-up in [`report/`](report/).

## Getting started

### 1. Clone

```bash
git clone https://github.com/<your-username>/stgnn-traffic-forecasting-metr-la.git
cd stgnn-traffic-forecasting-metr-la
```

### 2. Install dependencies

A GPU is recommended (developed on a Colab T4). Install PyTorch first, then the
graph packages that must match your PyTorch version:

```bash
pip install torch
export TORCH=$(python -c "import torch; print(torch.__version__)")
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install -r requirements.txt
```

On Google Colab, the first notebook cell installs everything for you.

### 3. Run

Open `notebooks/stgnns_metr_la.ipynb` in Jupyter or Colab and run the cells in
order. The notebook trains TTS, GWN and AGCRN in turn and saves figures to the
working directory. METR-LA is downloaded automatically on first run.

## Outputs

- Adjacency heatmaps (TTS full, GWN first 50 nodes)
- Average MAE / MAPE / MSE bar charts per model
- Actual vs predicted plots for the first three sensors
- `top_15_influential_nodes.csv`, the most influential GWN nodes and the nodes they affect most
- TensorBoard logs in `logs/` (`tensorboard --logdir logs`)



## Author
Nana Kwabena Opoku-Acheampong
MSc Artificial Intelligence, University of Cape Town
[LinkedIn](https://linkedin.com/in/nana-kwabena-opokuacheampong-623981237)
