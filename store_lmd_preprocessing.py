from pathlib import Path
import pandas as pd
from preprocessing.LMD import lmd_features
import numpy as np
from tqdm import tqdm
import pickle

from seiz_eeg.dataset import EEGDataset

# Make sure we catch overflows and all other errors
np.seterr(all='raise')

data_path = "data"

DATA_ROOT = Path(data_path)

clips_tr = pd.read_parquet(DATA_ROOT / "train/segments.parquet")
clips_te = pd.read_parquet(DATA_ROOT / "test/segments.parquet")

dataset_tr = EEGDataset(
    clips_tr,
    signals_root=DATA_ROOT / "train",
    prefetch=False,  # If your compute does not allow it, you can use `prefetch=False`
    signal_transform=lmd_features,
    return_id=True
)

dataset_te = EEGDataset(
    clips_te,  # Your test clips variable
    signals_root=DATA_ROOT / "test",  # Update this path if your test signals are stored elsewhere
    signal_transform=lmd_features,  # You can change or remove the signal_transform as needed
    prefetch=False,  # Set to False if prefetching causes memory issues on your compute environment
    return_id=True,  # Return the id of each sample instead of the label
)

with open('tmp/lmd_train_arrays.npy', 'rb') as f:
    train_arrays = np.load(f)
with open("tmp/lmd_train_ids.pickle", "rb") as f:
    train_ids = pickle.load(f)
    train_error_index = [i for i, (_, e) in enumerate(train_ids) if e is not None]

for i in train_error_index:
    try:
        x, y = dataset_tr[i]
        train_arrays[i] = x
        train_ids[i] = (y, None)
    except Exception as e:
        y = "_".join(map(str, clips_tr.index[i]))
        print(f"Error processing train sample {i} ({y}): {e}")
        train_ids[i] = (y, e)

    with open('tmp/lmd_train_arrays.npy', 'wb') as f:
        np.save(f, train_arrays)
    with open("tmp/lmd_train_ids.pickle", "wb") as f:
        pickle.dump(train_ids, f)
    print(f"Processed {i}/{len(clips_tr)} train samples")

with open('tmp/lmd_train_arrays.npy', 'wb') as f:
    np.save(f, train_arrays)
with open("tmp/lmd_train_ids.pickle", "wb") as f:
    pickle.dump(train_ids, f)

with open('tmp/lmd_test_arrays.npy', 'rb') as f:
    test_arrays = np.load(f)
with open("tmp/lmd_test_ids.pickle", "rb") as f:
    test_ids = pickle.load(f)
    test_error_index = [i for i, (_, e) in enumerate(test_ids) if e is not None]

for i in test_error_index:
    try:
        x, y = dataset_te[i]
        test_arrays[i] = x
        test_ids[i] = (y, None)
    except Exception as e:
        y = "_".join(map(str, clips_te.index[i]))
        print(f"Error processing test sample {i} ({y}): {e}")
        test_ids[i] = (y, e)

    with open('tmp/lmd_test_arrays.npy', 'wb') as f:
        np.save(f, test_arrays)
    with open("tmp/lmd_test_ids.pickle", "wb") as f:
        pickle.dump(test_ids, f)
    print(f"Processed {i}/{len(clips_te)} test samples")
    
with open('tmp/lmd_test_arrays.npy', 'wb') as f:
    np.save(f, test_arrays)
with open("tmp/lmd_test_ids.pickle", "wb") as f:
    pickle.dump(test_ids, f)