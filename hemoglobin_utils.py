import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
mne.set_log_level("ERROR")  # or "CRITICAL"

def load_hemoglobin_data(ID: int, chroma: str = "hbo", low_freq: float = 0.02, high_freq: float = 0.20):
    """
    Inputs:
    - subject_id (int): BIDS subject ID.
    - chroma (str): "hbo" or "hbr". (HbR = deoxy-hemoglobin, HbO = oxy-hemoglobin) 
    - l_freq, h_freq (float): Band-pass filter cutoffs (Hz).

    Outputs:
    - X (T, C): Hemoglobin normalized data (time x channels), ready for PCA/ICA.
    - time (T,): Time vector (s).
    - regressor (T,): Task ON/OFF boxcar regressor.
    - channel_names: Channel names.
    """

    # ERROR definitions
    subjects = [356, 362, 364, 365, 366, 371, 372, 375, 376, 378, 380]
    if ID not in subjects:
        raise ValueError(f"ID not found. Valid IDs: {subjects}")
    if chroma not in ("hbo", "hbr"):
        raise ValueError("chroma must be 'hbo' or 'hbr'")

    # ===============================================  LOAD and Transform snirf DATA  ================================================
    # Paths to SNIRF amd events file (BIDS structure)
    snirf_path = (
    f"../data/sub-{ID}/ses-1/nirs/"
    f"sub-{ID}_ses-1_task-FingerTapping_run-01_nirs.snirf")
    events_path = f"../data/sub-{ID}/ses-1/nirs/sub-{ID}_ses-1_task-FingerTapping_run-01_events.tsv"

    # Load + convert to Hemoglobin (Hb) 
    raw = mne.io.read_raw_snirf(snirf_path, preload=False, verbose="error")
    raw_optical_density = mne.preprocessing.nirs.optical_density(raw)   # OD = -log(I / I0)
    raw_hemoglobin = mne.preprocessing.nirs.beer_lambert_law(raw_optical_density)  #Applies Modified Beer–Lambert Law to estimate ΔHbO and ΔHbR
    raw_hemoglobin.pick(picks=[chroma])   

    # Apply Filter
    raw_hemoglobin.filter(l_freq=low_freq, h_freq=high_freq, verbose="error")

    # Extract data: MNE gives X = (C, T); transpose to X = (T, C) 
    data_hemoglobin = raw_hemoglobin.get_data().T 
    time = raw_hemoglobin.times

    # Normalize data
    data_mean = np.mean(data_hemoglobin, axis = 0, keepdims = True)
    data_std = np.std(data_hemoglobin, axis= 0, keepdims = True) + 1e-12
    X = (data_hemoglobin - data_mean) / data_std

    # =================================================   CREAT REGRESSOR FUNCTION =========================================================
    # Build boxcar regressor
    df_events = pd.read_csv(events_path, sep="\t").sort_values("onset")
    regressor = np.zeros_like(time, dtype=float)    # Initialize s(t)=0
    for onset, dur in zip(df_events["onset"].to_numpy(), df_events["duration"].to_numpy()):
        regressor[(time >= onset) & (time < onset + dur)] = 1.0 # Set s(t)=1 during each block

    # Useful metadata
    channel_names = np.array(raw_hemoglobin.ch_names)

    return X, time, regressor, channel_names
