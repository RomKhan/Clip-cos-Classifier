from torch.utils.data import Dataset
import os
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root_dir, database, transform=None, max_offers=None):
        self.root_dir = root_dir
        self.transform = transform
        self.max_offers = max_offers
        self.database = database
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        image_paths = []
        offers_count = 0
        for root, dirs, files in os.walk(self.root_dir):
            if self.max_offers is not None and offers_count >= self.max_offers:
                break
            if self.database.select_multiple(os.path.basename(root), 'offer') is not None:
                continue
            is_offer_folder = 0
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_paths.append((os.path.basename(root), os.path.join(root, file)))
                    is_offer_folder = 1
            offers_count += is_offer_folder
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        offer_id, image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, image_path, offer_id
