# fNIRS Finger Tapping — PCA & ICA (HbO / HbR)

End-to-end mini-project using **fNIRS finger tapping data** to decompose hemodynamic signals into latent components with **unsupervised learning**.  
The goal is to identify components correlated with task timing and components dominated by physiological or structured noise, using **Principal Component Analysis (PCA)** and **Independent Component Analysis (ICA)**, and to compare both methods quantitatively.




## Project goal

- Load fNIRS hemoglobin signals (**HbO / HbR**) from multiple subjects
- Apply minimal preprocessing (band-pass filtering)
- Decompose signals using:
  - **PCA** (variance-based components)
  - **ICA (FastICA)** (statistically independent components)
- Identify components that correlate with the finger-tapping task



## Data & preprocessing

- **Task:** Finger tapping
- **Signals:** HbO (oxy-hemoglobin) and HbR (deoxy-hemoglobin)
- **Subjects:**  
  `356, 362, 364, 365, 366, 371, 372, 375, 376, 378, 380`
- **Preprocessing:**
  - Conversion to hemoglobin signals
  - Band-pass filter: **0.02 – 0.20 Hz**
    - Removes slow drift and high-frequency physiological noise



## Methods

### PCA
- Applied to data matrix `X ∈ ℝ^{T×C}` (time × channels)
- Outputs:
  - Temporal scores `Z(t, k)`
  - Spatial weights `W(k, c)`
- Properties:
  - Orthogonal components
  - Maximizes explained variance
- Metrics:
  - **Reconstruction error:**  
    $\frac{\|X - \hat{X}\|}{\|X\|}$
    
  - **Task alignment:**  
    $max\_k |corr(Z_k(t), regressor(t))|$

### ICA (FastICA)
- Same input data as PCA
- Extracts statistically independent components
- Better suited to isolating structured noise and non-Gaussian sources
- Same metrics as PCA for direct comparison



## Results

### Console outputs

**PCA (HbR)**
- Subjects processed: 356, 362, 364, 365, 366, 371, 372, 375, 376, 378, 380
- Reconstruction error range: ~0.42 – 0.58
- Best task correlation (|corr|) observed: up to ~0.40
- For each subject:
  - PCs ranked by |corr| with task regressor
  - Explained variance ratio per PC

**ICA (HbO)**
- Reconstruction error range: ~0.23 – 0.35
- Best task correlation (|corr|) observed: up to ~0.34
- FastICA convergence warnings appear for some subjects

**ICA (HbR)**
- Reconstruction error range: ~0.42 – 0.58
- Best task correlation (|corr|) observed: up to ~0.39
- FastICA convergence warnings appear for some subjects



### Files generated

**CSV**
- `results/metrics_by_subject.csv`  
  Columns:
  - `subject`
  - `chrom` (hbo / hbr)
  - `method` (PCA / ICA)
  - `recon_error`
  - `best_abs_corr`
  - `best_component`
  - `best_corr`

## Dataset

This project uses the publicly available **Electrical_Thermal_FingerTapping_2015** dataset.

- **Modality:** functional Near-Infrared Spectroscopy (fNIRS)
- **Task:** Finger tapping
- **Format:** BIDS (version 1.7.1)
- **Signals analyzed:** HbO and HbR
- **License:** CC0

**Authors:**  
Yücel, Meryem; Selb, Juliette; Aasted, Christopher; Petkov, Mihayl;  
Borsook, David; Boas, David; Becerra, Lino.

**Ethics approval:**  
Massachusetts General Hospital IRB.


