import os
from data.base_dataset import BaseDataSet
import cv2 as cv 

class CarvanaDataset(Dataset):
    """Carvana Dataset"""
    def __init__(self, image_dir, mask_dir, transform=None) -> None:
        super(Dataset, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)  # List of image names in directory
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):  # Return a specific image and corresponding target at index
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))

        image = cv.imread(image_path, cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        cap = cv.VideoCapture(mask_path)
        _, mask = cap.read()
        cap.release()
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY).astype('float32')

        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask
