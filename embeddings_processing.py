from imageDataset import ImageDataset
import clip
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from torch import nn
from torchvision.models import ResNeXt101_64X4D_Weights, resnext101_64x4d
import numpy as np
from scipy.sparse import lil_matrix
from hdf5_work import save_embeddings, get_dataset_count


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

    return probs, image_paths, offers_idx, image_embeddings

def get_resnext_embeddings(device, dataset):
    model = resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    dataset.transform = ResNeXt101_64X4D_Weights.IMAGENET1K_V1.transforms()
    dataloader = DataLoader(dataset, batch_size=128)

    # image_embeddings = np.zeros((len(dataset), 2048), dtype='float32')
    image_embeddings = lil_matrix((len(dataset), 2048), dtype='float32')
    counter = 0
    for images, batch_image_paths, batch_offers_idx in tqdm(dataloader):
        with torch.no_grad():
            image_embeddings[counter: counter+len(images)] = nn.functional.normalize(model(images.to(device))).to('cpu').to(torch.float32).numpy()
            counter += len(images)


    return image_embeddings


def process_new_images(device, images_path, max_offers, db, embeddings_path):
    dataset = ImageDataset(root_dir=images_path, database=db, max_offers=max_offers)
    print(f'dataset consist of {len(dataset)} elements')
    print('getting clip embeddings:')
    probs, image_paths, offers_idx, clip_image_embeddings = get_clip_embeddings(device, dataset)
    print('getting resnext embeddings:')
    resnext_image_embeddings = get_resnext_embeddings(device, dataset)

    image_count = get_dataset_count(embeddings_path, 'embeddings')
    for i in range(0, len(image_paths), 50000):
        start = i
        end = i + 50000
        resnext_image_embeddings_temp = resnext_image_embeddings[start:end].toarray()
        save_embeddings(embeddings_path[start:end],
                        clip_image_embeddings[start:end],
                        resnext_image_embeddings_temp,
                        image_paths[start:end],
                        probs[start:end])

    offers_images = {}
    for i in range(len(offers_idx)):
        if offers_idx[i] not in offers_images:
            offers_images[offers_idx[i]] = []
        offers_images[offers_idx[i]].append(i+image_count)

    offers = []
    for key in offers_images:
        images_id = np.array(offers_images[key])
        offer_id = key
        offers.append((offer_id, images_id))
    db.insert_offers(offers)