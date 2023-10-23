import os
import sys
import h5py

from database import Database
from imageDataset import ImageDataset
import clip
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from torch import nn
from torchvision.models import ResNeXt101_64X4D_Weights, resnext101_64x4d
import numpy as np


def get_clip_embeddings(device, dataset):
    model, preprocess = clip.load('ViT-L/14', device=device)
    dataset.transform = preprocess
    dataloader = DataLoader(dataset, batch_size=128)
    text = clip.tokenize(["indoor", "room", "kitchen", "bedroom", "dining room", "living room", "lobby",
                          "concierge area", "elevator", "stairwell", "vestibule", "outdoor", "building",
                          "park", "tree", "grass", "road", "floor plan", "blueprint", "underground parking"]).to(device)

    probs = []
    image_paths = []
    offers_idx = []
    image_embeddings = np.zeros((len(dataset), 768), dtype='float32')
    counter = 0

    for images, batch_image_paths, batch_offers_idx in tqdm(dataloader):
        with torch.no_grad():
            image_embeddings[counter: counter+len(images)] = nn.functional.normalize(model.encode_image(images.to(device))).to('cpu').to(torch.float32).numpy()
            counter += len(images)
            logits_per_image, logits_per_text = model(images.to(device), text)
            batch_probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            probs.extend(batch_probs)
            image_paths.extend(batch_image_paths)
            offers_idx.extend(batch_offers_idx)

    image_embeddings = np.concatenate(image_embeddings, axis=0)

    return probs, image_paths, offers_idx, image_embeddings


def get_resnext_embeddings(device, dataset):
    model = resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    dataset.transform = ResNeXt101_64X4D_Weights.IMAGENET1K_V1.transforms()
    dataloader = DataLoader(dataset, batch_size=128)

    image_embeddings = np.zeros((len(dataset), 2048), dtype='float32')
    counter = 0
    for images, batch_image_paths, batch_offers_idx in tqdm(dataloader):
        with torch.no_grad():
            image_embeddings[counter: counter+len(images)] = nn.functional.normalize(model(images.to(device))).to('cpu').to(torch.float32).numpy()
            counter += len(images)


    return image_embeddings


def process_new_images(device, images_path, max_offers, db):
    dataset = ImageDataset(root_dir=images_path, database=db, max_offers=max_offers)
    print(f'dataset consist of {len(dataset)} elements')
    print('getting clip embeddings:')
    probs, image_paths, offers_idx, clip_image_embeddings = get_clip_embeddings(device, dataset)
    print('getting resnext embeddings:')
    resnext_image_embeddings = get_resnext_embeddings(device, dataset)

    offers_images = {}
    for i in range(len(offers_idx)):
        if offers_idx[i] not in offers_images:
            offers_images[offers_idx[i]] = []
        image = (clip_image_embeddings[i], resnext_image_embeddings[i], probs[i], image_paths[i])
        offers_images[offers_idx[i]].append(image)

    offers = []
    for key in offers_images:
        images_id = db.insert_offer_images(offers_images[key])
        offer_id = key
        offers.append((offer_id, images_id))
    db.insert_offers(offers)

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

def get_resnext_relevants(stacked_image_embeddings, relevant_idx_clip):
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


def create_dataset(probs, relevant_idx, relevant_idx_resnext):
    new_probs = np.zeros_like(probs)
    used_idx = []
    for i in tqdm(range(len(probs))):
        if i not in relevant_idx:
            continue
        used_idx.append(i)
        new_probs[i] = 0.85 * probs[i]
        idx_clip = relevant_idx[i][0].astype(np.int32)
        idx_resnext = relevant_idx_resnext[i][0].astype(np.int32)[:100]
        if len(idx_clip) > 0:
            softmax_clip = softmax(1 - relevant_idx[i][1], temperature=0.013)
            softmax_resnext = softmax(1 - relevant_idx_resnext[i][1][:100], temperature=0.013)
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

def save_target(path_to_dataset, used_clip_embeddings, target, idx, paths):
    full_path = os.path.join(path_to_dataset, 'target.hdf5')
    if not os.path.exists(full_path):
        with h5py.File(full_path, 'w') as f:
            f.create_dataset('clip features', data=used_clip_embeddings, compression="gzip", maxshape=(None,None))
            f.create_dataset('target', data=target, dtype='i1', maxshape=(None,))
            f.create_dataset('idx', data=idx, dtype='i8', maxshape=(None,))
            asciiList = np.stack([n.encode("ascii", "ignore") for n in paths], axis=0)
            dt = h5py.string_dtype(encoding='ascii')
            f.create_dataset('paths', chunks=True, dtype=dt, data=asciiList, maxshape=(None,))
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
            asciiList = np.stack([n.encode("ascii", "ignore") for n in paths], axis=0)
            paths_data[prev_len:] = asciiList
            idx_data[prev_len:] = idx


if __name__ == '__main__':
    device = sys.argv[1]
    images_path = sys.argv[2]
    max_offers = sys.argv[3]
    if max_offers == '-':
        max_offers = None
    path_to_dataset = sys.argv[4]
    if path_to_dataset == '-':
        path_to_dataset = ''
    path_to_relevants = sys.argv[5]
    if path_to_relevants == '-':
        path_to_relevants = ''
    db_path = ''
    if len(sys.argv) > 6:
        db_path = sys.argv[6]

    db = Database(db_path)
    if os.path.exists(os.path.join(path_to_dataset, 'target.hdf5')):
        with h5py.File(os.path.join(path_to_dataset, 'target.hdf5'), 'r') as f:
            prev_image_count = f['idx'][-1]
    else:
        prev_image_count = 0
    process_new_images(device, images_path, int(max_offers), db)
    current_image_count = db.get_images_count()

    while current_image_count >= prev_image_count:
        i_start = current_image_count - 1000000
        i_end = current_image_count
        i, clip_embeddings, resnext_embeddings, probs, paths = zip(*db.select_with_condition('image', i_start, i_end))
        clip_embeddings = np.stack(list(clip_embeddings), axis=0)
        resnext_embeddings = np.stack(list(resnext_embeddings), axis=0)
        probs = np.stack(list(probs))
        paths = np.array(list(paths))
        target_idx = range(min(clip_embeddings.shape[0]-(current_image_count - prev_image_count), 1000000), clip_embeddings.shape[0])
        print('getting relevants for clip embeddings')
        clip_relevants = get_clip_relevants(clip_embeddings, target_idx)
        print('getting relevants for resnext embeddings')
        resnext_relevants = get_resnext_relevants(resnext_embeddings, clip_relevants)
        Database.save_relevants(path_to_relevants, clip_relevants, resnext_relevants)
        del resnext_embeddings

        print('creating dataset')
        idx, target = zip(*create_dataset(probs, clip_relevants, resnext_relevants))
        idx = np.array(list(idx))
        target = np.array(list(target))
        del clip_relevants
        del resnext_relevants
        del probs

        save_target(path_to_dataset, clip_embeddings[idx], target, idx+1, paths[idx])
        current_image_count -= 1000000


















