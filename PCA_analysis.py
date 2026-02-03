import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hemoglobin_utils import load_hemoglobin_data
from sklearn.decomposition import PCA
from results_io import append_row


# Subjects ID
subjects = [356, 362, 364, 365, 366, 371, 372, 375, 376, 378, 380]

# Path to save results as csv
CSV_PATH = "./results/metrics_by_subject.csv"

# Parameters
low_frequency = 0.02  # Band-pass filter cutoffs (Hz).
high_frequency = 0.20  # Band-pass filter cutoffs (Hz).
chroma = "hbo"  # hbo/hbr  (HbR = deoxy-hemoglobin, HbO = oxy-hemoglobin)  BEST IS HBO
print(f"PCA Analysis Settings:\n"f"  Subjects: {subjects}\n"f"  Chromophore: {chroma}\n"f"  Band-pass filter: [{low_frequency}, {high_frequency}] Hz\n"f"  Output CSV: {CSV_PATH}")


# Cycle over IDs 
for id in subjects:
    #Load data
    X, time, regressor, channel_names =  load_hemoglobin_data(id, chroma, low_frequency, high_frequency)

    # Print subject id
    print(f"\n--PCA Analysis Results Subject-{id}--")

    # PCA
    K = 10  # number of components to inspect
    pca = PCA(n_components=K, random_state=0)

    # Fit PCA on X with shape (T, C)
    Z = pca.fit_transform(X)      # (T, K) temporal component scores
    W = pca.components_           # (K, C) spatial weights per component

    # Calculate error in signal reconstruction
    X_hat = Z @ W + pca.mean_
    error = np.linalg.norm(X - X_hat) / np.linalg.norm(X)
    print(f"Reconstruction error: {error:.3f}")

    # Explained variance 
    evar = pca.explained_variance_ratio_
    cum_evar = np.cumsum(evar)
    
    # Calculate stimulus correlation with regressor
    stimulus_correlations = []
    for i in range(K):
        corr = np.corrcoef(Z[:,i], regressor)[0,1]
        stimulus_correlations.append(corr)

    # Sort PCs per best stimulus correlation and print
    idx = np.argsort(np.abs(stimulus_correlations))[::-1]
    for j in idx[:K]:
        print(f"PC{j+1:>2d} : |corr|={abs(stimulus_correlations[j]):6.3f} | EVR={evar[j]:6.3f}")

    # Save PC data with best stimulus correlation
    best_j = idx[0] 
    best_corr = stimulus_correlations[best_j]

    append_row(CSV_PATH, {
        "subject": int(id),
        "chrom": chroma,
        "method": "PCA",
        "recon_error": float(error),
        "best_abs_corr": float(abs(best_corr)),
        "best_component": f"PC{best_j+1}",
        "best_corr": float(best_corr),
    })

        
    # Plot cumulative sum per subject
    plt.plot(cum_evar, marker="o", label = f"id={id}")
plt.title(f"Cumulative Explained Variance (k = {K})")
plt.grid()
plt.axhline(0.8, ls="--", c="k")
plt.xlabel("PC index")
plt.ylabel("Sum")
plt.legend(ncol=2)
plt.savefig(f"./results/PCA_cumulative_variance_{chroma}.png", dpi = 150)
plt.show() 

# Plot top 2 PCs per |corr| (only last subject)
for k in idx[:2]:
    plt.plot(time, Z[:, k], label=f"PC{k+1}")
plt.plot(time, regressor * Z[:, idx[0]].max(), "--", label="Task")
plt.xlabel("Time (s)")
plt.title(f"Subject {id}: Task-aligned PCs")
plt.legend()
plt.grid()
plt.savefig(f"./results/PCA_topPCs_task_time_sub-{id}_{chroma}.png", dpi = 150)
plt.show()

# Plot top 2 PCs weights per |corr|  (only last subject)
for i, k in enumerate(idx[:4]):
    plt.subplot(2,2,i+1)
    plt.stem(W[k])
    plt.title(f"PC{k+1} spatial weights")
    plt.grid()
    plt.xlabel("Channel index")
    plt.ylabel("Weight")
plt.suptitle(f"Subject {id}: PCA spatial maps (top PCs)")
plt.tight_layout()
plt.savefig(f"./results/PCA_topPCs_spatial_weights_sub-{id}_{chroma}.png", dpi = 150)
plt.show()








        