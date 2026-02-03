import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

# SANITY CHECK FOR TEMPORAL DESCRIPTION EVENTS (.tsv)

# Subjects ID
subjects = [356, 362, 364, 365, 366, 371, 372, 375, 376, 378, 380]

# Cycle over IDs 
for id in subjects:

    # Select path acording to ID
    sub = f"sub-{id}/ses-1/nirs/sub-{id}_ses-1_task-FingerTapping_run-01_events.tsv"

    # Load BIDS events.tsv
    path = f"../../data/{sub}"
    events = pd.read_csv(path, sep="\t")

    # Basic sanity checks
    print(f"\n=== Subject {id} ===")
    print(f"dim = {events.shape}")
    print(events)
    print(events.dtypes)
    
    # Build a simple boxcar regressor s(t)
    # representing "task ON/OFF" to test alignment with the stimulus.
    assert "onset" in events.columns, "Expected an 'onset' column in events.tsv"
    assert "duration" in events.columns, "Expected a 'duration' column in events.tsv"

    # Choose temporary sampling rate (Hz) just for testing
    fs_dummy = 50.0  

    # Define a time grid that covers the full task (until last event ends)
    t_end = float((events["onset"] + events["duration"]).max())
    t = np.arange(0.0, t_end, 1.0 / fs_dummy)

    # Initialize regressor with zeros
    s = np.zeros_like(t)

    # Fill in ones during each event interval [onset, onset+duration)
    for onset, dur in zip(events["onset"].values, events["duration"].values):
        onset = float(onset)
        dur = float(dur)
        mask = (t >= onset) & (t < onset + dur)
        s[mask] = 1.0

    print("Regressor shape:", s.shape)
    print(f"Time range: {t[0]} {t[-1]:.2f}")
    print(f"Task ON fraction: {s.mean():.2f}")

    plt.plot(t, s, "--",  label=f"sub-{id}")

plt.xlabel("Time (s)")
plt.ylabel("Task state")
plt.title("Finger tapping ON/OFF (all subjects, design check)")
plt.grid(True)
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig("finger_tapping_time_all_subjects.png", dpi = 150)
plt.show()