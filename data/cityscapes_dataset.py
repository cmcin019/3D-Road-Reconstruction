import os
from data.base_dataset import BaseDataSet
import cv2 as cv 

class CityscapesDataset(BaseDataSet):
    """Cityscapes Dataset"""
    def __init__(self, image_dir, mask_dir, transform=None):
        super(Dataset, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return self.images
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('leftImg8bit', 'gtFine_color'))
        instances_path = os.path.join(self.mask_dir, self.images[index].replace('leftImg8bit', 'instanceIds'))

        image = cv.imread(image_path, cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        mask = cv.imread(mask_path, cv.IMREAD_COLOR)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)

        instances = cv.imread(instances_path, cv.IMREAD_COLOR)
        instances = cv.cvtColor(instances, cv.COLOR_BGR2RGB)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask, instances=instances)
            image = augmentations['image']
            mask = augmentations['mask']
            instances = augmentations['instances']

        # return image, mask, instances
        return image, mask
