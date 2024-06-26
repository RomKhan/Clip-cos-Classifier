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
from hdf5_work import save_filter_embeddings, get_dataset_count

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
    is_test_mode = sys.argv[6]
    if is_test_mode == '-':
        is_test_mode = False
    else:
        is_test_mode = True
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
    model.eval()
    for keys, embeddings, paths in tqdm(dataloader):
        with torch.no_grad():
            preds = model(embeddings.to(device))
            preds = preds.detach().cpu()
            preds, current_classes = torch.max(F.softmax(preds, -1), dim=1)
            temp.extend(paths)
            target.extend(current_classes.tolist())
    model.train()
    target = np.array(target)

    if is_test_mode:
        folder_path = f'{path_to_filter_embeddings}/test_output'
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        if not os.path.exists(os.path.join(folder_path, 'indoor')):
            os.mkdir(os.path.join(folder_path, 'indoor'))
        if not os.path.exists(os.path.join(folder_path, 'others')):
            os.mkdir(os.path.join(folder_path, 'others'))

        for i in range(len(target)):
            key, embedding, image_path = images_dataset.keys[i], images_dataset.embeddings[i], images_dataset.paths[i]
            image_path = image_path.decode("ascii")
            label = target[i]
            destination_path = os.path.join(folder_path, 'indoor' if int(label) == 1 else 'others',
                                            '_'.join(image_path.split('/')[-3:]))
            shutil.copyfile(image_path, destination_path)

    indoor_idx = np.where(target == 1)[0]
    prev_image_count = get_dataset_count(path_to_filter_embeddings, 'filter_embeddings', 'paths')
    save_filter_embeddings(path_to_filter_embeddings,
                           images_dataset.embeddings[indoor_idx],
                           images_dataset.paths[indoor_idx])

    offers_images = {}
    for i in range(len(indoor_idx)):
        key = images_dataset.keys[indoor_idx[i]]
        if key not in offers_images:
            offers_images[key] = []
        offers_images[key].append(prev_image_count + i)

    offers = []
    for key in offers_images:
        images_id = np.array(offers_images[key])
        offer_id = key
        offers.append((offer_id, images_id))
    out_db.insert_offers(offers)






