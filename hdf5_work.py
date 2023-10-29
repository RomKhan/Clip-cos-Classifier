import os
import numpy as np
import h5py


def save_target(path_to_dataset, used_clip_embeddings, target, idx, paths):
    full_path = os.path.join(path_to_dataset, 'target.hdf5')
    if not os.path.exists(full_path):
        with h5py.File(full_path, 'w') as f:
            f.create_dataset('clip features', data=used_clip_embeddings, compression="gzip", maxshape=(None, None))
            f.create_dataset('target', data=target, dtype='i1', maxshape=(None,))
            f.create_dataset('idx', data=idx, dtype='i8', maxshape=(None,))
            dt = h5py.string_dtype(encoding='ascii')
            f.create_dataset('paths', chunks=True, dtype=dt, data=paths, maxshape=(None,))
    else:
        with h5py.File(full_path, 'r+') as f:
            x_data = f['clip features']
            y_data = f['target']
            idx_data = f['idx']
            paths_data = f['paths']
            prev_len = len(x_data)
            x_data.resize((prev_len + len(idx), used_clip_embeddings.shape[-1]))
            y_data.resize((prev_len + len(idx),))
            paths_data.resize((prev_len + len(idx),))
            idx_data.resize((prev_len + len(idx),))
            x_data[prev_len:] = used_clip_embeddings
            y_data[prev_len:] = target
            paths_data[prev_len:] = paths
            idx_data[prev_len:] = idx


def save_clip_embeddings(path_to_dataset, clip_embeddings, paths, probs):
    full_path = os.path.join(path_to_dataset, 'embeddings.hdf5')
    if not os.path.exists(full_path):
        with h5py.File(full_path, 'w') as f:
            f.create_dataset('clip embeddings', data=clip_embeddings, compression="gzip", maxshape=(None, None),
                             dtype='f4')
            f.create_dataset('probs', data=probs, maxshape=(None, None), dtype='f4')
            asciiList = np.stack([n.encode("ascii", "ignore") for n in paths], axis=0)
            dt = h5py.string_dtype(encoding='ascii')
            f.create_dataset('paths', chunks=True, dtype=dt, data=asciiList, maxshape=(None,))
    else:
        with h5py.File(full_path, 'r+') as f:
            clip_data = f['clip embeddings']
            paths_data = f['paths']
            probs_data = f['probs']
            prev_len = len(clip_data)
            clip_data.resize((prev_len + len(paths), clip_embeddings.shape[-1]))
            probs_data.resize((prev_len + len(paths), len(probs[0])))
            paths_data.resize((prev_len + len(paths),))
            clip_data[prev_len:] = clip_embeddings
            probs_data[prev_len:] = probs
            asciiList = np.stack([n.encode("ascii", "ignore") for n in paths], axis=0)
            paths_data[prev_len:] = asciiList


def save_other_model_embeddings(path_to_dataset, embeddings, model_name):
    full_path = os.path.join(path_to_dataset, 'embeddings.hdf5')
    with h5py.File(full_path, 'r+') as f:
        key = f'{model_name} embeddings'
        if key not in f.keys():
            f.create_dataset(key, data=embeddings, compression="gzip", maxshape=(None, None), dtype='f4')
        else:
            data = f[key]
            prev_len = len(data)
            data.resize((prev_len + embeddings.shape[0], embeddings.shape[-1]))
            data[prev_len:] = embeddings


def save_filter_embeddings(path_to_dataset, clip_embeddings, paths):
    full_path = os.path.join(path_to_dataset, 'filter_embeddings.hdf5')
    if not os.path.exists(full_path):
        with h5py.File(full_path, 'w') as f:
            f.create_dataset('clip embeddings', data=clip_embeddings, compression="gzip", maxshape=(None, None),
                             dtype='f4')
            dt = h5py.string_dtype(encoding='ascii')
            f.create_dataset('paths', chunks=True, dtype=dt, data=paths, maxshape=(None,))
    else:
        with h5py.File(full_path, 'r+') as f:
            clip_data = f['clip embeddings']
            paths_data = f['paths']
            prev_len = len(clip_data)
            clip_data.resize((prev_len + len(paths), clip_embeddings.shape[-1]))
            paths_data.resize((prev_len + len(paths),))
            clip_data[prev_len:] = clip_embeddings
            paths_data[prev_len:] = paths

def get_dataset_count(path_to_dataset, name, column_name = None, column_target=None):
    if os.path.exists(os.path.join(path_to_dataset, f'{name}.hdf5')):
        with h5py.File(os.path.join(path_to_dataset, f'{name}.hdf5'), 'r') as f:
            if column_target is not None:
                count = f[column_target][-1] + 1
            elif column_name != None:
                if column_name in f.keys():
                    count = len(f[column_name])
                else:
                    count = 0
            else:
                count = len(f[list(f.keys())[0]])
    else:
        count = 0
    return count
