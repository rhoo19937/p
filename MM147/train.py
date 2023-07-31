# https://github.com/terry-u16/ahc018
# ありがとうございます。


from typing import List, Tuple

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms.functional
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm


IMAGE_SIZE = 24
DATA_COUNT = 10000
DATA_DIR = f"data/data_{IMAGE_SIZE}"


def try_mkdir(path: str):
    if not os.path.isdir(path):
        os.mkdir(path)


#========================DatasetCode========================

class ImageDataset(Dataset):
    def __init__(self,indices: np.ndarray,path_x0: str,path_x1: str,path_y: str,):
        self.path_x0 = path_x0
        self.path_x1 = path_x1
        self.path_y = path_y
        self.indices = indices


    def __getitem__(self, i):
        index = self.indices[i]
        img_x0_path = f"{self.path_x0}/{index:0>4}.npy"
        img_x0 = np.load(img_x0_path)
        img_x0 = img_x0[np.newaxis, :, :]
        img_x0 = torch.from_numpy(img_x0.astype(np.float32)).clone()

        img_x1_path = f"{self.path_x1}/{index:0>4}.npy"
        img_x1 = np.load(img_x1_path)
        img_x1 = img_x1[np.newaxis, :, :]
        img_x1 = torch.from_numpy(img_x1.astype(np.float32)).clone()
        img_x = torch.cat([img_x0, img_x1])

        img_y_path = f"{self.path_y}/{index:0>4}.npy"
        img_y = np.load(img_y_path)
        img_y = img_y[np.newaxis, :, :]
        img_y = torch.from_numpy(img_y.astype(np.float32)).clone()

        data = {"x": img_x, "y": img_y}
        return data


    def __len__(self):
        return len(self.indices)




#========================ReadDataCode========================

def read_image(path: str) -> np.ndarray:
    with open(path) as f:
        array = np.array(list(map(float,f.read().split())))
        array.resize((IMAGE_SIZE,IMAGE_SIZE))

    return array


def write_cost_numpy(path: str, img: np.ndarray):
    np.save(path, img)


try_mkdir(f"{DATA_DIR}/numpy_x0")
try_mkdir(f"{DATA_DIR}/numpy_x1")
try_mkdir(f"{DATA_DIR}/numpy_y")

for seed in tqdm(range(10000)):
    ans_img = read_image(f"{DATA_DIR}/ans/{seed:0>4}.txt")
    sampled_img = read_image(f"{DATA_DIR}/sampled/{seed:0>4}.txt")
    flag_img = read_image(f"{DATA_DIR}/flag/{seed:0>4}.txt")
    write_cost_numpy(f"{DATA_DIR}/numpy_x0/{seed:0>4}.npy", sampled_img)
    write_cost_numpy(f"{DATA_DIR}/numpy_x1/{seed:0>4}.npy", flag_img)
    write_cost_numpy(f"{DATA_DIR}/numpy_y/{seed:0>4}.npy", ans_img)



#========================ModelCode========================

class TwoConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.rl = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rl(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rl(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.rl = nn.ReLU()

    def forward(self, x):
        x = self.up(x)
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.rl(x)
        return x


C=16 # todo

class UNet_2D(nn.Module):
    def __init__(self):
        super().__init__()
        # two-convs
        self.TCB1 = TwoConvBlock(2, C)
        self.TCB2 = TwoConvBlock(C, C*2)
        self.TCB3 = TwoConvBlock(C*2, C*4)

        self.TCB4 = TwoConvBlock(C*4, C*8)

        self.TCB5 = TwoConvBlock(C*8, C*4)
        self.TCB6 = TwoConvBlock(C*4, C*2)
        self.TCB7 = TwoConvBlock(C*2, C)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # up-convs
        self.UC1 = UpConv(C*8, C*4)
        self.UC2 = UpConv(C*4, C*2)
        self.UC3 = UpConv(C*2, C)

        self.conv1 = nn.Conv2d(C, 1, kernel_size=3, padding="same")
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # encoder
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)

        x = self.UC1(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB5(x)

        x = self.UC2(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC3(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB7(x)

        x = self.conv1(x)
        x = self.sigmoid(x)

        return x




#========================TrainCode========================

def setup_train_val_split(count: int) -> Tuple[np.ndarray, np.ndarray]:
    train_count = int(round(count * 0.8))
    train_indices = np.arange(train_count)
    val_indices = np.arange(train_count, count)

    return train_indices, val_indices


def setup_train_val_datasets() -> Tuple[ImageDataset, ImageDataset]:
    train_indices, val_indices = setup_train_val_split(10000)

    image_x0_path = f"{DATA_DIR}/numpy_x0"
    image_x1_path = f"{DATA_DIR}/numpy_x1"
    image_y_path = f"{DATA_DIR}/numpy_y"
    train_dataset = ImageDataset(train_indices, image_x0_path, image_x1_path, image_y_path)
    val_dataset = ImageDataset(val_indices, image_x0_path, image_x1_path, image_y_path)

    return train_dataset, val_dataset


def setup_train_val_loaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = setup_train_val_datasets()
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)

    return train_loader, val_loader


def train_1epoch(model: nn.Module, train_loader: DataLoader, lossfun, optimizer, lr_scheduler) -> float:
    model.train()
    total_loss = 0.0

    for data in tqdm(train_loader):
        x = data["x"]
        y = data["y"]

        optimizer.zero_grad()
        out = model(x)
        loss = lossfun(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        lr_scheduler.step()

    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss


def validate_1epoch(model: nn.Module, val_loader: DataLoader, lossfun) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data in tqdm(val_loader):
            x = data["x"]
            y = data["y"]

            out = model(x)
            loss = lossfun(out.detach(), y)

            total_loss += loss.item() * x.size(0)

    avg_loss = total_loss / len(val_loader.dataset)
    return avg_loss


def train(model: nn.Module, optimizer,train_loader: DataLoader, val_loader: DataLoader, lr_scheduler,n_epochs: int, ):
    lossfun = torch.nn.MSELoss()

    for epoch in range(n_epochs):
        train_loss = train_1epoch(
            model, train_loader, lossfun, optimizer, lr_scheduler
        )
        val_loss = validate_1epoch(model, val_loader, lossfun)

        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch={epoch}, train loss={train_loss}, val loss={val_loss}, lr={lr}")


def predict(model, loader):
    preds = []
    for data in tqdm(loader):
        with torch.set_grad_enabled(False):
            x = data["x"]
            y = model(x)

        y = y.cpu().numpy()
        for i in range(len(y)):
            preds.append(y[i, 0, :, :])

    return preds


def train_unet(batch_size: int) -> nn.Module:
    model = UNet_2D()
    n_epochs = 20
    train_loader, val_loader = setup_train_val_loaders(batch_size)
    n_iterations = len(train_loader) * n_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.001) # todo
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iterations)
    train(model, optimizer, train_loader, val_loader, n_epochs=n_epochs, lr_scheduler=lr_scheduler)
    return model


def predict_unet(model: nn.Module, batch_size: int) -> List[np.ndarray]:
    _, val_loader = setup_train_val_loaders(batch_size)
    preds = predict(model, val_loader)
    return preds


def run():
    batch_size = 16 # todo
    model = train_unet(batch_size)
    preds = predict_unet(model, batch_size)

    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()

    print(f"params: {params}")

    torch.save(model.state_dict(), f"{DATA_DIR}/model_weight.pth")
    try_mkdir(f"{DATA_DIR}/pred")

    for i, pred in enumerate(preds):
        pred = np.round(pred * 255)
        img = Image.fromarray(pred.astype(np.uint8))
        img.save(f"{DATA_DIR}/pred/{i:0>4}.bmp")



run()



#========================WeightExportCode========================

import base64
import struct
from collections import OrderedDict


def pack(value: np.float32) -> bytes:
    return struct.pack('<f', value)


def export_weights(weights: OrderedDict, path: str):
    with open(path, mode="w") as f:
        f.write("dict!(")
        for key, value in weights.items():
            value = value.flatten().to("cpu").numpy()
            stream = b""

            for v in value:
                stream += pack(v)

            s = base64.b64encode(stream).decode("utf-8")
            f.write(f"\"{key}\" => b\"{s}\",\n")
        f.write(")")


dict = torch.load(f"{DATA_DIR}/model_weight.pth")
export_weights(dict, f"{DATA_DIR}/weight_base64.txt")




#========================TestCode========================

import matplotlib.pyplot as plt
model = UNet_2D()
model.load_state_dict(torch.load(f"{DATA_DIR}/model_weight.pth"))
_, val_loader = setup_train_val_datasets()
data=val_loader[39]
plt.subplot(2,2,1)
plt.imshow(data["x"][0,:,:])
plt.subplot(2,2,2)
plt.imshow(data["x"][1,:,:])
plt.subplot(2,2,3)
x=data["x"]
x=x.resize_(1,2,IMAGE_SIZE,IMAGE_SIZE)
y=model(x)
plt.imshow(y.detach().numpy()[0,0,:,:])
plt.subplot(2,2,4)
plt.imshow(data["y"][0,:,:])
plt.show()