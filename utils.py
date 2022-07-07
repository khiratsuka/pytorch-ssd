# utils.py

import os
import requests
import time

from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

#import qmodels


class ImagenetMiniDataset(Dataset):
    def __init__(self, dataset_folder='./images',
                 class_idx={},
                 transforms=transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ]),
                 is_train=True):
        self.dataset_folder = dataset_folder
        self.class_idx = class_idx
        self.transforms = transforms
        self.is_train = is_train

        self.dataset_path, self.correct_class = self._get_file_names()


    def __getitem__(self, idx):
        dataset_path, correct_class = self.dataset_path[idx], self.correct_class[idx]
        
        #画像読み込み
        img = Image.open(dataset_path).convert('RGB')

        #Imagenetの値を使って正規化
        img = self.transforms(img)
        return img, correct_class


    def __len__(self):
        return len(self.dataset_path)


    def _get_file_names(self):
        phase = 'train' if self.is_train else 'val'
        
        #画像データのパスと正解クラスを入れるリスト
        img_path, correct_class, = [], []

        #クラス名のみ取得
        class_keys = list(self.class_idx.keys())
        for class_key in class_keys:
            img_dir = os.path.join(self.dataset_folder, phase, class_key)
            if not os.path.exists(img_dir):
                continue
            data_names = sorted([name for name in os.listdir(img_dir) if name.endswith('JPEG')])

            #フォルダ内全ての画像をチェック
            for data_name in data_names:
                data_name_path = os.path.join(img_dir, data_name)
                if not os.path.join(data_name_path):
                    continue

                img_path.append(data_name_path)
                correct_class.append(self.class_idx[class_key])

        assert len(img_path) == len(correct_class), '画像データの数とクラスの数が一致しません。'
        return list(img_path), list(correct_class)


def model_download(dir='./weights', url='https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'):
    if not os.path.exists(dir):
        os.makedirs(dir)
    filename = os.path.join(dir, os.path.basename(url))
    req_file = requests.get(url, stream=True)
    total_filesize = int(req_file.headers.get('content-length', 0))
    chunk_filesize = 32 * 1024
    downloaded_filesize = 0

    dl_progressbar = tqdm(desc=filename, total=total_filesize, unit='iB', unit_scale=True)
    with open(filename, 'wb') as f:
        for data in req_file.iter_content(chunk_filesize):
            f.write(data)
            downloaded_filesize += len(data)
            dl_progressbar.update(chunk_filesize)
    
    if total_filesize != 0 and downloaded_filesize == total_filesize:
        return False
    else:
        return True


def load_images(dir='./images', is_train=True):
    phase = 'train' if is_train else 'val'
    images_root_dir = os.path.join(dir, phase)

    #transformsの設定, 参照: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
    transform_settings = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    #ImageFolderの生成
    imageset = ImageFolder(root=images_root_dir, transform=transform_settings)
    
    return imageset


def calibration(model, dataloader):
    model.eval()
    loop_n = 0
    with tqdm(total=len(dataloader), unit='batch', desc='calibration') as prog_bar:
        with torch.no_grad():
            for image, target in dataloader:
                if loop_n > 100:
                    break
                model.forward(image)
                loop_n += 1
                prog_bar.update(1)


#https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html からの引用
def get_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size = (os.path.getsize("temp.p")/1e6)
    os.remove("temp.p")
    return size


def evaluate(model, device, dataloader, description):
    preds_acc = 0.0
    total_time = 0.0
    loop_n = 0
    with tqdm(total=len(dataloader), unit='batch', desc=description) as prog_bar:
        with torch.no_grad():
            for img, label in dataloader:
                loop_n += 1
                #画像データ、ラベルを指定したデバイスへ転送
                img = img.to(device)
                label = label.to(device)

                #推論の実行
                start_time = time.perf_counter()
                preds = model(img)
                layer_softmax = nn.Softmax(dim=1)
                batch_preds = layer_softmax(preds)
                end_time = time.perf_counter()

                #時間計測して記録
                elapsed_time = end_time - start_time
                total_time += elapsed_time

                #確率、ラベルを高い順にソート、Top-1を抽出して正誤をチェック
                _, preds_label = batch_preds.sort(dim=1, descending=True)
                for img_n in range(len(img)):
                    if preds_label[img_n][0] == label[img_n]:
                        preds_acc += 1.0
                prog_bar.update(1)
                
                if description == 'Prepare Model' and loop_n > 100:
                    break

    preds_acc = preds_acc / len(dataloader)
    return total_time, preds_acc

"""
#https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#helper-functions から引用
def load_model(model_file):
    model = qmodels.MobileNetV2()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model
"""