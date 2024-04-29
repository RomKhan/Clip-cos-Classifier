import os
import sys
import h5py

from database import Database
from tqdm import tqdm
import numpy as np
from hdf5_work import save_target, get_dataset_count

def get_clip_relevants(stacked_image_embeddings, target_idx):
    banned_idx = set()
    scip_count = 0
    relevant_idx = {}
    for i in tqdm(target_idx):
        if i in banned_idx:
            scip_count += 1
            continue
        cosine_distances = 1.0 - stacked_image_embeddings[i].dot(stacked_image_embeddings.T)
        same_pics = np.where(cosine_distances <= 0.01)[0].tolist()
        banned_idx.update(same_pics)

        condition = (cosine_distances < 0.17) & (cosine_distances > 0.01)
        idx = np.where(condition)[0]
        sorted_idx = idx[np.argsort(cosine_distances[idx])[:1000]]
        values = np.sort(cosine_distances[idx])[:1000]
        relevant_idx[i] = np.stack((sorted_idx, values)).astype('float32')
    return relevant_idx

def get_other_model_relevants(stacked_image_embeddings, relevant_idx_clip):
    relevant_idx = {}
    for i in tqdm(relevant_idx_clip):
        cosine_distances = 1.0 - stacked_image_embeddings[i].dot(stacked_image_embeddings.T)
        condition = (cosine_distances < 0.45) & (cosine_distances > 0.04)
        idx = np.where(condition)[0]
        sorted_idx = idx[np.argsort(cosine_distances[idx])[:100]]
        values = np.sort(cosine_distances[idx])[:100]
        relevant_idx[i] = np.stack((sorted_idx, values)).astype('float32')
    return relevant_idx


def softmax(x, temperature=1.0):
    return np.exp(x / temperature) / np.exp(x / temperature).sum()


def create_dataset_with_other_model(probs, relevant_idx, other_model_idx_resnext):
    new_probs = np.zeros_like(probs)
    used_idx = []
    for i in tqdm(range(len(probs))):
        if i not in relevant_idx:
            continue
        used_idx.append(i)
        new_probs[i] = 0.85 * probs[i]
        idx_clip = relevant_idx[i][0].astype(np.int32)
        idx_resnext = other_model_idx_resnext[i][0].astype(np.int32)[:100]
        if len(idx_clip) > 0:
            softmax_clip = softmax(1 - relevant_idx[i][1], temperature=0.013)
            softmax_resnext = softmax(1 - other_model_idx_resnext[i][1][:100], temperature=0.013)
            neighbor_clip_probs = softmax_clip @ probs[idx_clip]
            neighbor_resnext_probs = softmax_resnext @ probs[idx_resnext[:100]]
            new_probs[i] = 0.4 * probs[i] + 0.4 * neighbor_clip_probs + 0.2 * neighbor_resnext_probs

    tagret = []
    new_probs = np.column_stack((new_probs[:, :6].sum(axis=1, keepdims=True),
                                 new_probs[:, 6:11].sum(axis=1, keepdims=True),
                                 new_probs[:, 11:17].sum(axis=1, keepdims=True),
                                 new_probs[:, 17:19].sum(axis=1, keepdims=True),
                                 new_probs[:, 19:].sum(axis=1, keepdims=True),
                                 ))
    predicted_class = np.argmax(new_probs, axis=1)[used_idx]
    probs = np.max(new_probs, axis=1)[used_idx]
    for i in tqdm(range(len(predicted_class))):
        if predicted_class[i] == 0 and probs[i] >= 0.8:
            tagret.append((used_idx[i], 1))
        elif predicted_class[i] == 1 and probs[i] >= 0.7:
            tagret.append((used_idx[i], 0))
        elif predicted_class[i] != 0 and predicted_class[i] != 1 and probs[i] >= 0.2:
            tagret.append((used_idx[i], 0))
    return tagret


def create_dataset(probs, relevant_idx):
    new_probs = np.zeros_like(probs)
    used_idx = []
    for i in tqdm(range(len(probs))):
        if i not in relevant_idx:
            continue
        used_idx.append(i)
        new_probs[i] = 0.85 * probs[i]
        idx_clip = relevant_idx[i][0].astype(np.int32)
        if len(idx_clip) > 0:
            softmax_clip = softmax(1 - relevant_idx[i][1], temperature=0.013)
            neighbor_clip_probs = softmax_clip @ probs[idx_clip]
            new_probs[i] = 0.5 * probs[i] + 0.5 * neighbor_clip_probs

    tagret = []
    new_probs = np.column_stack((new_probs[:, :6].sum(axis=1, keepdims=True),
                                 new_probs[:, 6:11].sum(axis=1, keepdims=True),
                                 new_probs[:, 11:17].sum(axis=1, keepdims=True),
                                 new_probs[:, 17:19].sum(axis=1, keepdims=True),
                                 new_probs[:, 19:].sum(axis=1, keepdims=True),
                                 ))
    predicted_class = np.argmax(new_probs, axis=1)[used_idx]
    probs = np.max(new_probs, axis=1)[used_idx]
    for i in tqdm(range(len(predicted_class))):
        if predicted_class[i] == 0 and probs[i] >= 0.8:
            tagret.append((used_idx[i], 1))
        elif predicted_class[i] == 1 and probs[i] >= 0.7:
            tagret.append((used_idx[i], 0))
        elif predicted_class[i] != 0 and predicted_class[i] != 1 and probs[i] >= 0.2:
            tagret.append((used_idx[i], 0))
    return tagret


if __name__ == '__main__':
    path_to_dataset = sys.argv[1]
    if path_to_dataset == '-':
        path_to_dataset = ''
    sim_model_name = sys.argv[2]
    if sim_model_name == '-':
        sim_model_name = None
    max_images = sys.argv[3]
    if max_images == '-':
        max_offers = None
    path_to_relevants = sys.argv[4]
    if path_to_relevants == '-':
        path_to_relevants = ''
    relevants_count = sys.argv[5]
    if relevants_count == '-':
        relevants_count = 400000
    elif relevants_count.isnumeric():
        relevants_count = int(relevants_count)
    else:
        print(f'{relevants_count} - не число')
        sys.exit(1)

    prev_image_count = get_dataset_count(path_to_dataset, 'target', 'idx')
    current_image_count = get_dataset_count(path_to_dataset, 'embeddings')

    if current_image_count < relevants_count:
        print('too small for relevants')
        sys.exit(1)

    while current_image_count >= prev_image_count:
        i_start = prev_image_count
        i_end = min(current_image_count, prev_image_count + 1000000)
        if i_end - i_start < relevants_count:
            i_start = max(current_image_count - relevants_count, 0)

        print('getting relevants for clip embeddings')
        with h5py.File(os.path.join(path_to_dataset, 'embeddings.hdf5'), 'r+') as f:
            clip_data = f['clip embeddings']
            probs_data = f['probs']
            paths_data = f['paths']
            clip_embeddings = clip_data[i_start:i_end]
            probs = probs_data[i_start:i_end]
            paths = paths_data[i_start:i_end]

        target_idx = range(min(clip_embeddings.shape[0]-(current_image_count - prev_image_count), 1000000), clip_embeddings.shape[0])
        clip_relevants = get_clip_relevants(clip_embeddings, target_idx)
        del clip_embeddings

        other_model_relevants = None
        if sim_model_name is not None:
            print(f'getting relevants for {sim_model_name} embeddings')
            with h5py.File(os.path.join(path_to_dataset, 'embeddings.hdf5'), 'r+') as f:
                sim_model_data = f[f'{sim_model_name} embeddings']
                sim_model_embeddings = sim_model_data[i_start:i_end]

            other_model_relevants = get_other_model_relevants(sim_model_embeddings, clip_relevants)

            #Database.save_relevants(path_to_relevants, clip_relevants, other_model_relevants)

            del sim_model_embeddings

        with h5py.File(os.path.join(path_to_dataset, 'embeddings.hdf5'), 'r+') as f:
            clip_data = f['clip embeddings']
            clip_embeddings = clip_data[i_start:i_end]

        print('creating dataset')
        if sim_model_name is None:
            idx, target = zip(*create_dataset(probs, clip_relevants))
        else:
            idx, target = zip(*create_dataset_with_other_model(probs, clip_relevants, other_model_relevants))
        idx = np.array(list(idx))
        target = np.array(list(target))
        del clip_relevants
        del other_model_relevants
        del probs

        save_target(path_to_dataset, clip_embeddings[idx], target, idx, paths[idx])
        prev_image_count += 1000000
