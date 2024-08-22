import os
import numpy as np
from PIL import Image
import torch
import json
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from imblearn.over_sampling import SMOTE
from tqdm import tqdm


def save_images(images, labels, class_mapping, output_dir):
    for cls_name in class_mapping.keys():
        os.makedirs(os.path.join(output_dir, cls_name), exist_ok=True)

    for i, (image, label) in tqdm(enumerate(zip(images, labels)),total=len(images)):
        class_name = list(class_mapping.keys())[list(class_mapping.values()).index(label)]
        output_path = os.path.join(output_dir, class_name, f"smote_{i}.png")
        image_pil = transforms.ToPILImage()(image)
        image_pil.save(output_path)


def map_classes_to_labels(classes_distribution, class_mapping):
    dct={}
    for _class, val in classes_distribution.items():
        dct[class_mapping[_class]] = int(val)
    return dct

class SmoteDataLoader(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load data and get label
        X = self.data[index][0]
        y = torch.tensor(int(self.data[index][1]))
        if self.transform:
            X = self.transform(X)
        return X, y

def read_training_data(folder,args):
    idx=0
    images = []
    labels = []
    mapping = {}
    subfolders = os.listdir(folder)
    subfolders = sorted(subfolders)
    for subfolder in subfolders:
        mapping[subfolder]=idx
        for file in os.listdir(os.path.join(folder,subfolder)):
            img=Image.open(os.path.join(folder,subfolder,file))
            H, W = args.input_size[0], args.input_size[1]
            img=img.resize((H,W))
            images.append(img)
            labels.append(np.array(idx))
        idx = idx + 1
    return np.array(images), np.array(labels), mapping

class OutofMemoryError(RuntimeError):
    pass

def apply_smote(args,train_transforms):
    try:
        images, labels, class_mapping = read_training_data(args.data_path+'train',args)  #train -> Training folder path
        images_flattened = images.reshape(images.shape[0], -1)
        classes_distribution = map_classes_to_labels(args.classes_distribution, class_mapping)
        smote = SMOTE(sampling_strategy = classes_distribution)
        features_resampled, labels_resampled = smote.fit_resample(images_flattened, labels)
        H, W = args.input_size[0], args.input_size[1]
        images_resampled = [Image.fromarray(feature.reshape((H,W,3))) for feature in features_resampled]
        resampled_data = list(zip(images_resampled, labels_resampled))
        augmented_dataset = SmoteDataLoader(resampled_data,transform=train_transforms)
        num_classes = len(class_mapping)
        return augmented_dataset, num_classes

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            message = (f"{str(e)} This error mostly happened due to SMOTE as SMOTE requires all the training data "
                "to be read in one array not in batches to work which may cause OutofMemoryError. "
                "Try to disable SMOTE or reduce the image sizes or try other oversampling techniques.")
            raise OutofMemoryError(message) from None
        else:
            raise RuntimeError(str(e)) from None
