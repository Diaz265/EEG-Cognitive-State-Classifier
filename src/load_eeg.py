import sys
import importlib

try:
    mne = importlib.import_module("mne")
except ImportError:
    print("mne is required to run this module. Install it with: pip install mne", file=sys.stderr)
    sys.exit(1)

import numpy as np

def load_eeg():
    files = mne.datasets.eegbci.load_data(1, [6, 10])

    X_list = []
    y_list = []

    for i, f in enumerate(files):
        raw = mne.io.read_raw_edf(f, preload=True, verbose=False)

        raw.pick_types(eeg=True)
        raw.filter(0.5, 45)

        data = raw.get_data()

        for start in range(0, data.shape[1] - 128, 128):
            segment = data[:, start:start+128]

            if segment.shape[0] >= 32:
                segment = segment[:32, :]
                X_list.append(segment)
                y_list.append(i % 2)

    X = np.array(X_list).astype(np.float32)
    y = np.array(y_list)

    return X, y