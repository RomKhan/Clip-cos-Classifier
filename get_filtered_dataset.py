import os
import shutil
import sys
import torch
from tqdm import tqdm
import torch.nn.functional as F
from database import Database
from torch.utils.data import DataLoader
from imageModel import ImageModel
import numpy as np

def collate(batch):
    keys, embeddings, paths = zip(*batch)
    return list(keys), torch.Tensor(list(embeddings)), list(paths)

if __name__ == '__main__':
    device = sys.argv[1]
    model_path = sys.argv[2]
    in_db_path = sys.argv[3]
    if in_db_path == '-':
        in_db_path = ''
    out_db_path = sys.argv[3]
    if out_db_path == '-':
        out_db_path = ''
    model = torch.load(model_path, map_location=device).to(device)
    in_db = Database(in_db_path)
    out_db = Database(out_db_path, mode='filter db')

    in_idx_dict = in_db.get_offer_idx()
    out_idx_dict = out_db.get_offer_idx()
    offer_idx_to_add = list(set(in_idx_dict) - set(out_idx_dict))
    image_idx = {}
    for idx in offer_idx_to_add:
        image_idx[idx] = in_idx_dict[idx]

    images_dataset = []
    for key in image_idx:
        for id in image_idx[key]:
            _, clip_embedding, _, _, path = in_db.select_multiple(int(id), 'image')
            images_dataset.append((key, clip_embedding, path))
    dataloader = DataLoader(images_dataset, collate_fn=collate, batch_size=128)


    target = []
    for keys, embeddings, paths in tqdm(dataloader):
        with torch.no_grad():
            preds = model(embeddings.to(device))
            preds = preds.detach().cpu()
            preds, current_classes = torch.max(F.softmax(preds, -1), dim=1)
            target.extend(current_classes.tolist())

    # os.mkdir(os.path.join('output', 'indoor'))
    # os.mkdir(os.path.join('output', 'others'))
    #
    # for i in range(len(target)):
    #     key, embedding, image_path = images_dataset[i]
    #     label = target[i]
    #     destination_path = os.path.join('output', 'indoor' if int(label) == 1 else 'others', '_'.join(image_path.split('/')[-3:]))
    #     shutil.copyfile(image_path, destination_path)

    offers_images = {}
    for i in range(len(target)):
        key, embedding, path = images_dataset[i]
        is_indoor = True if target[i] == 1 else 0
        if is_indoor:
            if key not in offers_images:
                offers_images[key] = []
            image = (embedding, path)
            offers_images[key].append(image)

    offers = []
    for key in offers_images:
        images_id = out_db.insert_offer_images(offers_images[key])
        offer_id = key
        offers.append((offer_id, images_id))
    out_db.insert_offers(offers)




