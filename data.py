import io
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.v2 as transforms


def loadDataset(split):
    # Load and preprocess the data
    splits = {
        "train": "train-00000-of-00001.parquet",
        "test": "test-00000-of-00001.parquet",
    }

    df = pd.read_parquet("mnist/" + splits[split])
    tensor_image = transforms.PILToTensor()

    dataset = []
    for idx, data in tqdm(df.iterrows(), desc=f"Loading {split} dataset", total=len(df)):
        dataset.append(
            (
                tensor_image(Image.open(io.BytesIO(data["image"]["bytes"])))
                * (1.0 / 255.0),
                torch.tensor(data["label"], dtype=torch.int64),
            )
        )

    return dataset
