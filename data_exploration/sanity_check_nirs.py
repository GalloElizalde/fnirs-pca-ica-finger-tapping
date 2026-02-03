import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

# SANITY CHECK FOR TEMPORAL DESCRIPTION EVENTS (.tsv)

# Subjects ID
subjects = [356 , 364, 365, 366, 371, 372, 375, 376, 378, 380, 362]

# Cycle over IDs 
for id in subjects:

    # Path to SNIRF file (BIDS structure)
    snirf_path = (
    f"../../data/sub-{id}/ses-1/nirs/"
    f"sub-{id}_ses-1_task-FingerTapping_run-01_nirs.snirf"
    )

    # Load SNIRF header + metadata (no full data in RAM)
    raw = mne.io.read_raw_snirf(snirf_path, preload=False, verbose="error")

    print(f"\n--- sub-{id} ---")

    # Sampling frequency (Hz)
    print("sfreq (Hz):", raw.info["sfreq"])

    # Total number of channels (wavelength-specific)
    print("nchan:", raw.info["nchan"])

    # Channel types (fnirs_cw_amplitude, etc.)
    print("channel types:", set(raw.get_channel_types()))

    # Preview channel naming convention
    print("first 10 ch names:", raw.ch_names[:10])

    # Number of time samples
    print("n_times:", raw.n_times)

    # Recording duration in seconds
    print("duration (s):", raw.times[-1])
    

# Extract the raw data matrix as a NumPy array
data = raw.get_data()
print(data.shape)   # (n_channels, n_samples)
times = raw.times   # Get the time vector in seconds (one value per sample)
print(times.shape)  # Should match the number of samples
names = raw.ch_names  # Get the list of channel names


# Plot the first 5 channels to visually inspect their time series
for i, name in enumerate(names):
    if i == 5:
        break  # Stop after plotting 5 channels
    plt.plot(times, data[i], label=name)

plt.title(f"sub-{id} FingerTapping run-01 (first 5 channels)")
plt.xlabel("Time")
plt.ylabel("Amplitude (raw units)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"./finger_tapping_channels_sub{id}.png", dpi=150)
plt.show()