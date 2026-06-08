import numpy as np

try:
    import mne
except ImportError as exc:
    raise ImportError("mne is required for EEG preprocessing. Install it with `pip install mne`.") from exc

def load_and_preprocess(files):
    all_epochs = []

    for f in files:
        raw = mne.io.read_raw_edf(f, preload=True, verbose=False)

        # 1. garder EEG uniquement
        raw.pick_types(eeg=True)

        # 2. filtrage cerveau (important)
        raw.filter(0.5, 45)

        # 3. normalisation simple
        data = raw.get_data()
        data = (data - np.mean(data)) / np.std(data)

        all_epochs.append(data)

    return np.array(all_epochs)


if __name__ == "__main__":
    files = mne.datasets.eegbci.load_data(1, [6, 10, 14])

    X = load_and_preprocess(files)

    print("Shape:", X.shape)