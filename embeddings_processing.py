import sys
from database import Database
from imageDataset import ImageDataset, ImageSimpleDataset
import clip
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from torch import nn
from torchvision.models import ResNeXt101_64X4D_Weights, resnext101_64x4d
from torchvision.models import ResNet152_Weights, resnet152
import numpy as np
from hdf5_work import save_clip_embeddings, save_other_model_embeddings, get_dataset_count


def get_clip_embeddings(device, embeddings_path, db_path, images_path, max_offers):
    db = Database(db_path)
    dataset = ImageDataset(root_dir=images_path, database=db, max_offers=max_offers)
    print(f'dataset consist of {len(dataset)} elements')
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

    model.eval()
    for images, batch_image_paths, batch_offers_idx in tqdm(dataloader):
        with torch.no_grad():
            image_embeddings[counter: counter+len(images)] = nn.functional.normalize(model.encode_image(images.to(device))).to('cpu').to(torch.float32).numpy()
            counter += len(images)
            logits_per_image, logits_per_text = model(images.to(device), text)
            batch_probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            probs.extend(batch_probs)
            image_paths.extend(batch_image_paths)
            offers_idx.extend(batch_offers_idx)

    image_count = get_dataset_count(embeddings_path, 'embeddings')
    save_clip_embeddings(embeddings_path, image_embeddings, image_paths, probs)

    offers_images = {}
    for i in range(len(offers_idx)):
        if offers_idx[i] not in offers_images:
            offers_images[offers_idx[i]] = []
        offers_images[offers_idx[i]].append(i + image_count)

    offers = []
    for key in offers_images:
        images_id = np.array(offers_images[key])
        offer_id = key
        offers.append((offer_id, images_id))
    db.insert_offers(offers)

def get_simple_model_embeddings(device, embeddings_path, model, model_name, max_image, transform):
    dataset = ImageSimpleDataset(embeddings_path, model_name, max_image, transform)
    print(f'dataset consist of {len(dataset)} elements')
    dataloader = DataLoader(dataset, batch_size=128)
    model.eval()

    image_embeddings = np.zeros((len(dataset), 2048), dtype='float32')
    counter = 0
    for images in tqdm(dataloader):
        with torch.no_grad():
            image_embeddings[counter: counter+len(images)] = nn.functional.normalize(model(images.to(device))).to('cpu').to(torch.float32).numpy()
            counter += len(images)

    save_other_model_embeddings(embeddings_path, image_embeddings, model_name)


def get_resnext():
    model = resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    transform = ResNeXt101_64X4D_Weights.IMAGENET1K_V1.transforms()
    return model, transform

def get_resnet():
    model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    transform = ResNet152_Weights.IMAGENET1K_V2.transforms()
    return model, transform

if __name__ == '__main__':
    device = sys.argv[1]
    model = sys.argv[2]
    max_offers = sys.argv[3]
    if max_offers == '-':
        max_offers = None
    path_to_dataset = sys.argv[4]
    if path_to_dataset == '-':
        path_to_dataset = ''
    images_path = ''
    if len(sys.argv) > 5:
        images_path = sys.argv[5]
    db_path = ''
    if len(sys.argv) > 6:
        db_path = sys.argv[6]

    if model == 'clip':
        print('getting clip embeddings:')
        get_clip_embeddings(device, path_to_dataset, db_path, images_path, int(max_offers))
    elif model == 'resnext':
        print('getting embeddings:')
        model, transform = get_resnext()
        get_simple_model_embeddings(device, path_to_dataset, model, 'resnext', 200000, transform)
    elif model == 'resnet':
        print('getting embeddings:')
        model, transform = get_resnet()
        get_simple_model_embeddings(device, path_to_dataset, model, 'resnet', 200000, transform)
