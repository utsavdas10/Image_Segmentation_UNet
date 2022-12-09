import os
from PIL import Image
from torch.utils.data import Dataset, dataloader
from torchvision import transforms, datasets


class CellDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        image = Image.open(img_path)
        mask = Image.open(mask_path)
        convert_tensor = transforms.ToTensor()
        image = convert_tensor(image)
        mask = convert_tensor(mask)
        mask[mask == 255] = 1

        return image, mask

