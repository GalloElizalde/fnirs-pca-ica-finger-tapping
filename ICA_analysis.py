import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hemoglobin_utils import load_hemoglobin_data
from sklearn.decomposition import FastICA
from results_io import append_row

# Subjects ID
subjects = [356, 362, 364, 365, 366, 371, 372, 375, 376, 378, 380]

# Path to save results as csv
CSV_PATH = "./results/metrics_by_subject.csv"

# Parameters
low_frequency = 0.02  # Band-pass filter cutoffs (Hz).
high_frequency = 0.20  # Band-pass filter cutoffs (Hz).
chroma = "hbr"  # hbo/hbr  (HbR = deoxy-hemoglobin, HbO = oxy-hemoglobin)
print(f"ICA Analysis Settings:\n"f"  Subjects: {subjects}\n"f"  Chromophore: {chroma}\n"f"  Band-pass filter: [{low_frequency}, {high_frequency}] Hz\n"f"  Output CSV: {CSV_PATH}")

# Cycle over IDs 
for id in subjects:

    #Load data
    X, time, regressor, channel_names =  load_hemoglobin_data(id, chroma, low_frequency, high_frequency)

    # Print subject id
    print(f"\n--ICA Analysis Results Subject-{id}--")

    # ICA Parameters
    K = 10  
    max_iter = 5000
    ica = FastICA(n_components=K, whiten="unit-variance", random_state=0, max_iter=max_iter, tol=1e-3)

    # Fit ICA on X
    S = ica.fit_transform(X)      # (T, K) independent component time courses
    A = ica.mixing_               # (C, K) spatial mixing weights per component

    # Reconstruct X from ICA 
    X_hat = S @ A.T + ica.mean_
    error = np.linalg.norm(X - X_hat) / np.linalg.norm(X)
    print(f"Reconstruction error: {error:.3f}")

    # Compute correlation of each IC time course with the task regressor (interpretation only)
    stimulus_correlations = []
    for k in range(K):
        corr = np.corrcoef(S[:, k], regressor)[0, 1]
        stimulus_correlations.append(corr)

    # Sort ICs by |correlation| and print
    idx = np.argsort(np.abs(stimulus_correlations))[::-1]
    for j in idx[:K]:
        print(f"IC{j+1:>2d} : |corr|={abs(stimulus_correlations[j]):6.3f} | corr={stimulus_correlations[j]:6.3f}")

    # Save ICA data with best stimulus correlation
    best_j = idx[0]
    best_corr = stimulus_correlations[best_j]

    append_row(CSV_PATH, {
        "subject": int(id),
        "chrom": chroma,
        "method": "ICA",
        "recon_error": float(error),
        "best_abs_corr": float(abs(best_corr)),
        "best_component": f"IC{best_j+1}",
        "best_corr": float(best_corr),
    })


# Plots for top 2 ICs (time courses) + regressor
for k in idx[:2]:
    plt.plot(time, S[:, k], label=f"IC{k+1}")
plt.plot(time, regressor * S[:, idx[0]].max(), "--", label="Task")
plt.title(f"Subject {id}: Task-aligned ICs")
plt.xlabel("Time (s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"./results/ICA_task_aligned_ICs_time_sub-{id}_{chroma}.png", dpi=150)
plt.show()


# Plot top 2 PCs weights per |corr|  (only last subject)
for i, k in enumerate(idx[:4]):
    plt.subplot(2,2,i+1)
    plt.stem(A[:, k])
    plt.title(f"IC{k+1} spatial weights")
    plt.xlabel("Channel index")
    plt.ylabel("Weight")
    plt.grid(True)

plt.suptitle(f"Subject {id}: ICA spatial maps (top ICs)")
plt.tight_layout()
plt.savefig(f"./results/ICA_topICs_spatial_weights_sub-{id}_{chroma}.png", dpi=150)
plt.show()