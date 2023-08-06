#!/user/bin/env python
from PIL import Image
import argparse
import torch
from torchvision.transforms import transforms
import pandas as pd
import numpy as np

class ThumbnailScoreDataset(torch.utils.data.Dataset):
    def __init__(self, df, images_path):

        self.df = df

        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Resize((224,224))
        ])

    def __len__(self):
        return len(self.df)
    
    def apply_transforms(self, x):
        x = self.transforms(x)
        x = x.float()
        return x
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        thumbnail_path = "/Users/joshua/Documents/life/media/ai_gen/DALL·E 2022-10-10 16.07.47 - A colorful print by Matisse of a robot head profile picture.png" # row['Thumbnail']
        view_count = row['View Count']

        thumbnail = Image.open(thumbnail_path).convert('RGB')
        thumbnail = self.apply_transforms(thumbnail)

        return thumbnail, view_count

def ThumbnailScoreDataloaders(csv_path: str, images_path:str , batch_size: int = 16, split_ratio: int = 0.9):

    # load dataset and shuffle it
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1)

    # split dataset into train and test sets
    mask = np.random.rand(len(df)) < split_ratio
    train_df, test_df = df[mask], df[~mask]
    assert (len(train_df) + len(test_df)) == len(df), "dataset split error"

    # define the datasets
    train_dataset = ThumbnailScoreDataset(df=train_df, images_path=images_path)
    test_dataset = ThumbnailScoreDataset(df=test_df, images_path=images_path)

    # define the dataloaders
    train_dl = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        )
    test_dl = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dl, test_dl

if __name__ == '__main__':

    # initialise arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--csv_path', type=str, required=True)
    parser.add_argument('-f', '--images_path', type=str, required=True)
    args = parser.parse_args()

    train_dl, test_dl = ThumbnailScoreDataloaders(args.csv_path, args.images_path)

    for batch in train_dl:
        images, scores = batch
        print(images.shape)
        print(scores.shape)