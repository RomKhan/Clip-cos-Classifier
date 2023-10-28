import os
import shutil
import sys

import h5py
import torch
from tqdm import tqdm
import torch.nn.functional as F
from database import Database
from torch.utils.data import DataLoader
from imageDataset import FilterImageDataset
from imageModel import ImageModel
import numpy as np
from hdf5_work import save_filter_embeddings

# def collate(batch):
#     keys, embeddings, paths = zip(*batch)
#     return list(keys), torch.Tensor(list(embeddings)), list(paths)

if __name__ == '__main__':
    device = sys.argv[1]
    model_path = sys.argv[2]
    in_db_path = sys.argv[3]
    if in_db_path == '-':
        in_db_path = ''
    out_db_path = sys.argv[3]
    if out_db_path == '-':
        out_db_path = ''
    embeddings_path = sys.argv[4]
    if embeddings_path == '-':
        embeddings_path = ''
    path_to_filter_embeddings = sys.argv[5]
    if path_to_filter_embeddings == '-':
        path_to_filter_embeddings = ''
    model = torch.load(model_path, map_location=device).to(device)
    in_db = Database(in_db_path)
    out_db = Database(out_db_path, mode='filter db')

    in_idx_dict = in_db.get_offer_idx()
    out_idx_dict = out_db.get_offer_idx()
    offer_idx_to_add = list(set(in_idx_dict) - set(out_idx_dict))
    image_idx = {}
    image_count = 0
    for idx in offer_idx_to_add:
        image_idx[idx] = in_idx_dict[idx]
        image_count += len(image_idx[idx])

    with h5py.File(os.path.join(embeddings_path, 'embeddings.hdf5'), 'r+') as f:
        clip_data = f['clip embeddings']
        paths_data = f['paths']
        keys = []
        embeddings = np.zeros((image_count, 768), dtype='float32')
        image_paths = []
        i = 0
        for key in image_idx:
            clip_embeddings, paths = clip_data[image_idx[key]], paths_data[image_idx[key]]
            keys.extend([key]*len(image_idx[key]))
            image_paths.extend(paths.tolist())
            embeddings[i:i+len(image_idx[key])] = clip_embeddings
            i += len(image_idx[key])
        image_paths = np.array(image_paths)
    images_dataset = FilterImageDataset(keys, embeddings, image_paths)
    dataloader = DataLoader(images_dataset, batch_size=128)

    target = []
    temp = []
    for keys, embeddings, paths in tqdm(dataloader):
        with torch.no_grad():
            preds = model(embeddings.to(device))
            preds = preds.detach().cpu()
            preds, current_classes = torch.max(F.softmax(preds, -1), dim=1)
            temp.extend(paths)
            target.extend(current_classes.tolist())
    target = np.array(target)

    # os.mkdir(os.path.join('output', 'indoor'))
    # os.mkdir(os.path.join('output', 'others'))
    #
    # for i in range(len(target)):
    #     key, embedding, image_path = images_dataset.keys[i], images_dataset.embeddings[i], images_dataset.paths[i]
    #     image_path = image_path.decode("ascii")
    #     label = target[i]
    #     destination_path = os.path.join('output', 'indoor' if int(label) == 1 else 'others',
    #                                     '_'.join(image_path.split('/')[-3:]))
    #     shutil.copyfile(image_path, destination_path)

    indoor_idx = np.where(target == 1)[0]
    save_filter_embeddings(path_to_filter_embeddings,
                           images_dataset.embeddings[indoor_idx],
                           images_dataset.paths[indoor_idx])

    offers_images = {}
    for i in range(len(indoor_idx)):
        key = images_dataset.keys[indoor_idx[i]]
        if key not in offers_images:
            offers_images[key] = []
        offers_images[key].append(i)

    os.mkdir(os.path.join('output', 'indoor_v2'))

    offers = []
    for key in offers_images:
        images_id = np.array(offers_images[key])
        offer_id = key
        offers.append((offer_id, images_id))
    out_db.insert_offers(offers)

    # with h5py.File(os.path.join(path_to_filter_embeddings, 'filter_embeddings.hdf5'), 'r+') as f:
    #     paths_data = f['paths']
    #     d = paths_data[:]
    #     for offer in offers:
    #         _, idx = offer
    #         paths = paths_data[idx]
    #         for path in paths:
    #             image_path = path.decode("ascii")
    #             destination_path = os.path.join('output', 'indoor_v2','_'.join(image_path.split('/')[-3:]))
    #             shutil.copyfile(image_path, destination_path)






