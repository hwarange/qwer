from src.data import MNISTDataset, MNISTDataModule
from src.model.auto_encoder import Encoder, Decoder, AutoEncoder
from src.training import MNISTModule

import numpy as np
import json
import random

from sklearn.model_selection import train_test_split

import torch
import torchvision.datasets as ds
import torchvision.transforms as transforms

import lightning as L
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

import seaborn as sns


def main(configs):
    # 'MNIST' 데이터셋을 로드 and preprocessing
    train = ds.MNIST(
        root='data/mnist',
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )

    test = ds.MNIST(
        root='data/mnist',
        train=False,
        transform=transforms.ToTensor(),
        download=True,
    )

    # 데이터셋을 학습용과 검증 데이터로 분할
    train, valid = train_test_split(
        train,
        test_size=0.2,
        random_state=configs.get('seed'),
    )

    train = [(X, y) for X, y in train if y == 1]
    valid = [(X, y) for X, y in valid if y == 1]

    # 데이터셋 객체로 변환
    train_dataset = MNISTDataset(train)
    valid_dataset = MNISTDataset(valid)
    test_dataset = MNISTDataset(test)

    # 데이터 모듈 생성 및 데이터 준비
    mnist_data_module = MNISTDataModule(batch_size=configs.get('batch_size'))
    mnist_data_module.prepare(train_dataset, valid_dataset, test_dataset)

    # 모델 생성
    configs.update({
        'input_dim': train[0][0].shape[1]*train[0][0].shape[2],
    })
    encoder = Encoder(configs.get('input_dim'), 5)
    decoder = Decoder(configs.get('input_dim'), 5)
    model = AutoEncoder(encoder, decoder)

    # LightningModule 인스턴스 생성
    mnist_module = MNISTModule(
        model=model,
        learning_rate=configs.get('learning_rate'),
    )

    # Trainer 인스턴스 생성 및 설정
    del configs['output_dim'], configs['seed'], configs['epochs']
    exp_name = ','.join([f'{key}={value}' for key, value in configs.items()])
    trainer_args = {
        'max_epochs': configs.get('epochs'),
        'callbacks': [
            EarlyStopping(monitor='loss/val_loss', mode='min', patience=10)
        ],
        'logger': TensorBoardLogger(
            'tensorboard',
            f'MNIST/{exp_name}',
        ),
    }

    if configs.get('device') == 'gpu':
        trainer_args.update({'accelerator': configs.get('device')})

    trainer = Trainer(**trainer_args)

    # 모델 학습 시작
    trainer.fit(
        model=mnist_module,
        datamodule=mnist_data_module,
    )


if __name__ == '__main__':
    # 사용 가능한 GPU가 있는 경우 'cuda', 그렇지 않으면 'cpu' 사용
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    # hyperparameter
    with open('./configs.json', 'r') as file:
        configs = json.load(file)
    configs.update({'device': device})

    # seed 설정
    seed = configs.get('seed')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA 설정
    if device == 'gpu':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    main(configs)
