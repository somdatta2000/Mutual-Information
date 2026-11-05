#!/usr/bin/env python3

"""
Mutual Information Analysis of Protein Conformational Dynamics
==============================================================

Calculates Mutual Information (MI) between two C-alpha distance
coordinates obtained from molecular dynamics simulations.

Coordinates:
    x(t): Tyr101-Asp289 distance
    y(t): Tyr101-Leu31 distance

Analysis:
    1. Replica-wise Mutual Information
    2. kNN-based MI estimation
    3. Block-shuffled null distribution
    4. Empirical p-value calculation
    5. Half-trajectory convergence analysis
    6. Pooled Mutual Information analysis
    7. Pearson correlation coefficient
    8. Joint probability distribution

Outputs:
    MI_dist_dist_summary_LABEL.csv
    MI_dist_dist_LABEL_5replicas.png

Author: Somdatta
"""

import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression


# ============================================================
# USER SETTINGS
# ============================================================

LABEL = "0mM"

REPLICAS = [1, 2, 3, 4, 5]

K = 3                 # kNN neighbours
NSHUFFLE = 500        # Number of block-shuffling iterations
NBINS2D = 40          # Bins per axis for P(x,y) histogram

# ============================================================


# ------------------------------------------------------------
# 1. Mutual Information Estimation
# ------------------------------------------------------------

def mi_bits(x, y):

    """
    Calculate Mutual Information in bits using
    k-nearest-neighbour estimation.
    """

    return mutual_info_regression(
        x[:, None],
        y,
        n_neighbors=K,
        random_state=0
    )[0] / np.log(2)


# ------------------------------------------------------------
# 2. Autocorrelation Time Estimation
# ------------------------------------------------------------

def act(a):

    """
    Estimate the autocorrelation decay time using
    the first crossing below 1/e.
    """

    a = a - a.mean()
    n = len(a)

    if a.std() == 0:
        return 1

    f = np.fft.rfft(a, 2 * n)

    acf = np.fft.irfft(
        f * np.conj(f)
    )[:n]

    acf /= acf[0]

    idx = np.where(acf < 1 / np.e)[0]

    return int(idx[0]) if len(idx) else n // 10


# ------------------------------------------------------------
# 3. Block Length Selection
# ------------------------------------------------------------

def pick_block(x, y):

    """
    Determine block length using the maximum
    autocorrelation decay time of x and y.
    """

    return int(
        min(
            max(2 * max(act(x), act(y)), 5),
            len(x) // 10
        )
    )


# ------------------------------------------------------------
# 4. Block Shuffling
# ------------------------------------------------------------

def shuffle_blocks(a, block, rng):

    """
    Randomly shuffle non-overlapping blocks while
    preserving the temporal structure within blocks.
    """

    n = len(a) // block * block

    return (
        a[:n]
        .reshape(-1, block)[rng.permutation(n // block)]
        .ravel()
    )


# ------------------------------------------------------------
# 5. Empirical P-value
# ------------------------------------------------------------

def pval(obs, null):

    """
    Calculate empirical p-value by comparing
    observed MI with the shuffled null distribution.
    """

    return (
        np.sum(null >= obs) + 1
    ) / (
        len(null) + 1
    )


# ============================================================
# LOAD INPUT DATA
# ============================================================

files = {
    n: "CA_dist_%s_rep%d.dat" % (LABEL, n)
    for n in REPLICAS
}

missing = [
    f for f in files.values()
    if not os.path.exists(f)
]

if missing:

    sys.exit(
        "Missing files: %s\n"
        "Run the CA-distance calculation script first for %s."
        % (missing, LABEL)
    )


# Load data: time, Tyr101-Asp289, Tyr101-Leu31

data = {
    n: np.loadtxt(files[n], unpack=True)
    for n in REPLICAS
}


# ============================================================
# REPLICA-WISE MUTUAL INFORMATION ANALYSIS
# ============================================================

rng = np.random.default_rng(0)

rows = []

for n in REPLICAS:

    t, x, y = data[n]

    # Determine block length
    block = pick_block(x, y)

    m = len(x) // block * block

    # Observed MI
    mi = mi_bits(
        x[:m],
        y[:m]
    )

    # Block-shuffled null distribution
    null = np.array([
        mi_bits(
            shuffle_blocks(x, block, rng),
            y[:m]
        )
        for _ in range(NSHUFFLE)
    ])

    # Half-trajectory analysis
    h = len(x) // 2

    h1 = mi_bits(
        x[:h],
        y[:h]
    )

    h2 = mi_bits(
        x[h:],
        y[h:]
    )

    # Pearson correlation
    r = np.corrcoef(x, y)[0, 1]

    # Empirical p-value
    p = pval(mi, null)

    rows.append(
        dict(
            name=str(n),
            n=len(x),
            block=block,
            mi=mi,
            null=null.mean(),
            p=p,
            h1=h1,
            h2=h2,
            r=r
        )
    )

    print(
        "rep%d: block=%d | MI=%.4f bits | "
        "null=%.4f | p=%.4f | halves %.4f / %.4f | r=%.3f"
        % (
            n,
            block,
            mi,
            null.mean(),
            p,
            h1,
            h2,
            r
        )
    )


# ============================================================
# POOLED MUTUAL INFORMATION ANALYSIS
# ============================================================

# Common block length: median across replicas

block_p = int(
    np.median([
        r["block"] for r in rows
    ])
)

xs = [
    data[n][1]
    for n in REPLICAS
]

ys = [
    data[n][2]
    for n in REPLICAS
]

# Trim each replica to complete blocks

trim = [
    len(a) // block_p * block_p
    for a in xs
]

X = np.concatenate([
    a[:m]
    for a, m in zip(xs, trim)
])

Y = np.concatenate([
    a[:m]
    for a, m in zip(ys, trim)
])


# Observed pooled MI

mi_pool = mi_bits(X, Y)


# Pooled shuffled null distribution

null_pool = np.array([

    mi_bits(
        np.concatenate([
            shuffle_blocks(a[:m], block_p, rng)
            for a, m in zip(xs, trim)
        ]),
        Y
    )

    for _ in range(NSHUFFLE)

])


# Pooled empirical p-value

p_pool = pval(
    mi_pool,
    null_pool
)


# Pooled half-trajectory MI

h1 = mi_bits(

    np.concatenate([
        a[:len(a)//2]
        for a in xs
    ]),

    np.concatenate([
        b[:len(b)//2]
        for b in ys
    ])

)

h2 = mi_bits(

    np.concatenate([
        a[len(a)//2:]
        for a in xs
    ]),

    np.concatenate([
        b[len(b)//2:]
        for b in ys
    ])

)


# Pooled Pearson correlation

r_pool = np.corrcoef(X, Y)[0, 1]


# Append pooled results

rows.append(

    dict(
        name="pooled",
        n=len(X),
        block=block_p,
        mi=mi_pool,
        null=null_pool.mean(),
        p=p_pool,
        h1=h1,
        h2=h2,
        r=r_pool
    )

)


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

hdr = (
    "replica,n_frames,block,MI_bits,null_mean_bits,"
    "p,MI_first_half,MI_second_half,pearson_r"
)

csv = "MI_dist_dist_summary_%s.csv" % LABEL

print("\n" + hdr)

with open(csv, "w") as f:

    f.write(hdr + "\n")

    for r in rows:

        line = (
            "%s,%d,%d,%.5f,%.5f,%.4f,%.5f,%.5f,%.4f"
            % (
                r["name"],
                r["n"],
                r["block"],
                r["mi"],
                r["null"],
                r["p"],
                r["h1"],
                r["h2"],
                r["r"]
            )
        )

        print(line)

        f.write(line + "\n")


# ============================================================
# VISUALIZATION
# ============================================================

fig, ax = plt.subplots(
    2,
    3,
    figsize=(15, 8),
    constrained_layout=True
)

cols = plt.cm.tab10(
    np.arange(len(REPLICAS))
)


# ------------------------------------------------------------
# Panel 1: Tyr101-Asp289 distance time series
# ------------------------------------------------------------

for c, n in zip(cols, REPLICAS):

    t, x, y = data[n]

    ax[0, 0].plot(
        t / 1000.0,
        x,
        lw=0.5,
        color=c,
        label="rep%d" % n
    )

ax[0, 0].set(
    xlabel="Time (ns)",
    ylabel="C$\\alpha$–C$\\alpha$ Tyr101–Asp289 (Å)",
    title="x(t)"
)

ax[0, 0].legend(
    fontsize=7,
    ncol=5
)


# ------------------------------------------------------------
# Panel 2: Tyr101-Leu31 distance time series
# ------------------------------------------------------------

for c, n in zip(cols, REPLICAS):

    t, x, y = data[n]

    ax[0, 1].plot(
        t / 1000.0,
        y,
        lw=0.5,
        color=c
    )

ax[0, 1].set(
    xlabel="Time (ns)",
    ylabel="C$\\alpha$–C$\\alpha$ Tyr101–Leu31 (Å)",
    title="y(t)"
)


# ------------------------------------------------------------
# Panel 3: Joint probability distribution
# ------------------------------------------------------------

hh = ax[0, 2].hist2d(
    X,
    Y,
    bins=NBINS2D,
    cmap="viridis",
    cmin=1
)

fig.colorbar(
    hh[3],
    ax=ax[0, 2],
    label="Frames"
)

ax[0, 2].set(
    xlabel="C$\\alpha$–C$\\alpha$ Tyr101–Asp289 (Å)",
    ylabel="C$\\alpha$–C$\\alpha$ Tyr101–Leu31 (Å)",
    title="P(x, y), pooled"
)


# ------------------------------------------------------------
# Panel 4: Replica-wise Mutual Information
# ------------------------------------------------------------

reps = rows[:-1]

ax[1, 0].bar(
    [r["name"] for r in reps],
    [r["mi"] for r in reps],
    color=cols
)

ax[1, 0].plot(
    [r["name"] for r in reps],
    [r["null"] for r in reps],
    "k_",
    ms=25,
    label="null mean"
)

ax[1, 0].set(
    xlabel="Replica",
    ylabel="MI (bits)",
    title="MI per replica"
)

ax[1, 0].legend()


# ------------------------------------------------------------
# Panel 5: Half-trajectory convergence
# ------------------------------------------------------------

idx = np.arange(len(reps))

w = 0.38

ax[1, 1].bar(
    idx - w/2,
    [r["h1"] for r in reps],
    w,
    label="first half",
    color="tab:cyan"
)

ax[1, 1].bar(
    idx + w/2,
    [r["h2"] for r in reps],
    w,
    label="second half",
    color="tab:orange"
)

ax[1, 1].set_xticks(idx)

ax[1, 1].set_xticklabels(
    [r["name"] for r in reps]
)

ax[1, 1].set(
    xlabel="Replica",
    ylabel="MI (bits)",
    title="Half-split convergence"
)

ax[1, 1].legend()


# ------------------------------------------------------------
# Panel 6: Pooled MI null distribution
# ------------------------------------------------------------

ax[1, 2].hist(
    null_pool,
    bins=30,
    color="grey",
    alpha=0.8,
    label="within-replica block-shuffled null"
)

ax[1, 2].axvline(
    mi_pool,
    color="crimson",
    lw=2,
    label="pooled = %.3f bits\np = %.3f"
    % (mi_pool, p_pool)
)

ax[1, 2].set(
    xlabel="MI (bits)",
    ylabel="Count",
    title="Pooled MI, %s" % LABEL
)

ax[1, 2].legend()


# ============================================================
# SAVE FIGURE
# ============================================================

out = "MI_dist_dist_%s_5replicas.png" % LABEL

plt.savefig(
    out,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\nAnalysis completed successfully.")

print("Saved figure:", out)
print("Saved summary:", csv)
