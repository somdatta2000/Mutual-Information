# Mutual Information Analysis of GPCR Conformational Dynamics

## Overview

This repository contains a Python-based workflow for estimating Mutual Information (MI) between two structural distance coordinates obtained from molecular dynamics (MD) simulations.

The analysis investigates the statistical dependence between two Cα–Cα distances associated with the Neuropeptide Y4 receptor (Y4R) and its bound pancreatic polypeptide (PP) ligand.

The two structural coordinates considered are:

- **x(t):** Tyr101–Asp289 Cα–Cα distance, associated with receptor activation.
- **y(t):** Tyr101–Leu31 Cα–Cα distance, associated with the PP-binding region.

Mutual Information is estimated using a k-nearest-neighbour (kNN) approach and expressed in bits.

The workflow includes replica-wise analysis, block-shuffled null distributions, half-trajectory convergence analysis and pooled Mutual Information estimation.

## Repository Structure

```text
GPCR-Mutual-Information-Analysis/
│
├── README.md
├── MI1.py
├── requirements.txt
│
├── input_data/
│   ├── CA_dist_0mM_rep1.dat
│   ├── CA_dist_0mM_rep2.dat
│   ├── CA_dist_0mM_rep3.dat
│   ├── CA_dist_0mM_rep4.dat
│   └── CA_dist_0mM_rep5.dat
│
└── output/
    ├── MI_dist_dist_0mM_5replicas.png
    └── MI_dist_dist_summary_0mM.csv
```

*Note: The input and output directories illustrate the recommended organization. The current script reads and writes files in its working directory.*

## Requirements

The analysis requires Python 3 and the following libraries:

- NumPy
- Matplotlib
- Scikit-learn

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Input Data

The analysis requires distance time-series data from five independent MD simulation replicas.

Each input file contains three columns:

| Column | Description |
|---|---|
| 1 | Simulation time (ps) |
| 2 | Tyr101–Asp289 Cα–Cα distance (Å) |
| 3 | Tyr101–Leu31 Cα–Cα distance (Å) |

### Input File Naming Convention

For the 0 mM serotonin condition:

```text
CA_dist_0mM_rep1.dat
CA_dist_0mM_rep2.dat
CA_dist_0mM_rep3.dat
CA_dist_0mM_rep4.dat
CA_dist_0mM_rep5.dat
```

The input files must be generated using the corresponding Cα-distance extraction procedure before running the MI analysis.

## Analysis Methodology

### 1. Mutual Information Estimation

Mutual Information is calculated using the k-nearest-neighbour estimator implemented in Scikit-learn's `mutual_info_regression`.

The number of nearest neighbours is set to:

```python
K = 3
```

The estimated MI is converted from nats to bits by dividing by ln(2).

### 2. Autocorrelation-Based Block Selection

The temporal autocorrelation functions of both distance coordinates are calculated using Fourier transforms.

The block length is determined using the first autocorrelation decay below 1/e.

The selected block length is used for subsequent block-shuffling procedures.

### 3. Block-Shuffled Null Distribution

To account for temporal correlations in MD trajectories, a block-shuffling procedure is employed.

Blocks of the Tyr101–Asp289 distance coordinate are randomly rearranged while the Tyr101–Leu31 distance coordinate remains unchanged.

The analysis performs:

```python
NSHUFFLE = 500
```

shuffling iterations to generate a null distribution of Mutual Information values.

An empirical p-value is calculated by comparing the observed MI against the shuffled null distribution.

### 4. Replica-Wise Analysis

Mutual Information is independently estimated for each of the five replicas.

The following parameters are reported:

- Number of trajectory frames.
- Selected block length.
- Observed Mutual Information (bits).
- Mean shuffled-null MI (bits).
- Empirical p-value.
- MI from the first half of the trajectory.
- MI from the second half of the trajectory.
- Pearson correlation coefficient.

### 5. Pooled Mutual Information Analysis

The distance coordinates from all five replicas are combined for pooled MI estimation.

A common block length is selected using the median block length across replicas.

The pooled analysis includes:

- Pooled observed MI.
- Pooled block-shuffled null distribution.
- Empirical p-value.
- First-half and second-half MI estimates.
- Pearson correlation coefficient.

## Execution

Modify the user settings in `MI1.py` according to the simulation condition:

```python
LABEL = "0mM"

REPLICAS = [1, 2, 3, 4, 5]

K = 3

NSHUFFLE = 500

NBINS2D = 40
```

Ensure that all five input files are present in the working directory.

Run the analysis using:

```bash
python MI1.py
```

## Output Files

The script generates two output files.

### 1. Mutual Information Summary

**Filename:**

```text
MI_dist_dist_summary_0mM.csv
```

The summary CSV contains replica-wise and pooled MI results.

The columns include:

| Parameter | Description |
|---|---|
| replica | Replica identifier |
| n_frames | Number of trajectory frames |
| block | Selected block length |
| MI_bits | Observed Mutual Information |
| null_mean_bits | Mean shuffled-null MI |
| p | Empirical p-value |
| MI_first_half | First-half MI |
| MI_second_half | Second-half MI |
| pearson_r | Pearson correlation coefficient |

The final row corresponds to the pooled analysis.

### 2. Mutual Information Visualization

**Filename:**

```text
MI_dist_dist_0mM_5replicas.png
```

The generated figure contains six panels:

**Panel 1: Tyr101–Asp289 distance time series**

Displays the temporal variation of the Tyr101–Asp289 distance across the five replicas.

**Panel 2: Tyr101–Leu31 distance time series**

Displays the temporal variation of the Tyr101–Leu31 distance across the five replicas.

**Panel 3: Joint probability distribution**

Shows the two-dimensional distribution P(x,y) of the pooled distance coordinates.

**Panel 4: Replica-wise Mutual Information**

Compares the observed MI values across replicas with their corresponding shuffled-null mean values.

**Panel 5: Half-trajectory convergence**

Compares Mutual Information estimated from the first and second halves of each replica.

**Panel 6: Pooled MI null distribution**

Displays the pooled shuffled-null MI distribution, observed pooled MI and corresponding empirical p-value.

## Applications

This workflow can be applied to investigate statistical dependencies between structural coordinates in GPCR molecular dynamics simulations.

In the context of Y4R, it provides a quantitative approach to examining the relationship between ligand-binding-associated structural fluctuations and receptor activation-associated conformational changes.

## Author

**Somdatta**

Department of Chemistry  
Indian Institute of Technology Jodhpur  
India

## Citation

If you use this workflow in your research, please cite this GitHub repository and the associated research publication when available.

## License

MIT License (if selected for this repository).
