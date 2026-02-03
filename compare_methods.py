import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./results/metrics_by_subject.csv")

# Promedios por método y cromóforo
print(df.groupby(["chrom","method"])[["recon_error","best_abs_corr"]].mean())

# Tabla wide por sujeto
wide = df.pivot(index="subject", columns=["chrom","method"], values=["recon_error","best_abs_corr"])
wide.to_csv("./results/compare_pca_ica_hbo_hbr.csv")
print("Saved: ./results/compare_pca_ica_hbo_hbr.csv")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10,4), sharey=True)
for ax, chrom in zip(axes, ["hbo", "hbr"]):
    sub = df[df["chrom"] == chrom]
    for method in ["PCA", "ICA"]:
        d = sub[sub["method"] == method]
        ax.scatter(d["best_abs_corr"], d["recon_error"], label=method)

    ax.set_title(chrom.upper())
    ax.set_xlabel("best |corr| with task")
    ax.grid(True)

axes[0].set_ylabel("reconstruction error")
axes[1].legend()
plt.suptitle("PCA vs ICA: task alignment vs reconstruction error")
plt.tight_layout()
plt.savefig("./results/compare_pca_ica_hbo_hbr.png", dpi=150)
plt.show()

