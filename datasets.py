from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import pandas as pd

class dr_dataset(Dataset):
    def __init__(self,
                 data: np.ndarray,
                 data_dir: str,
                 transform: None,
                 target_transform: None,
                 ext :str = ".png"):
        """
        Constructor for Diabetic Retinopathy Dataset Class.
        :param data: Numpy Array consisting of (img_name, class) entries
        :param data_dir: Directory in which the images are stored
        :param transform: Transformation that is to be applied on the images
        :param target_transform: Transformation that is to be applied to the labels
        :param ext: Extension of the images, in case the extension is not applied in data
        """
        self.data = pd.DataFrame(data)
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.ext = ext


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path = os.path.join(self.data_dir, self.data.iloc[item, 0]+self.ext )
        img = Image.open(img_path)
        label = self.data.iloc[item, 1]

        if self.transform:
            img = self.transform(img)
        else:
            self.transform = transforms.Compose([
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        return (img, label,)