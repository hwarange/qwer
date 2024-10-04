from src.data import IMDBDataset, IMDBDataModule
from src.model.lstm import Model
from src.training import IMDBModule

import pandas as pd
import numpy as np
import random
import json
from tqdm import tqdm

from tensorflow.keras.datasets import imdb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

import torch

import lightning as L
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

import seaborn as sns


def main(configs):
    # 'imdb' 데이터셋 로드
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
    
    # preprocessing
    # 200보다 길이가 긴 문장은 200개로 잘라서 사용
    max_len = 200
    x_train = [torch.Tensor(sequence[:max_len]).long() for sequence in x_train]
    x_test = [torch.Tensor(sequence[:max_len]).long() for sequence in x_test]

    # max_len보다 작은 문장을 max_len으로 맞춤
    x_train = pad_sequence(x_train, batch_first=True)
    x_test = pad_sequence(x_test, batch_first=True)

    # 데이터셋을 학습용과 임시 데이터로 분할
    # test 데이터를 검증용과 테스트용 데이터로 분할
    x_valid, x_test = train_test_split(
        x_test,
        test_size=0.5,
        random_state=seed,
    )
    y_valid, y_test = train_test_split(
        y_test,
        test_size=0.5,
        random_state=seed,
    )

    # 데이터셋 객체로 변환
    train_dataset = IMDBDataset(x_train, y_train)
    valid_dataset = IMDBDataset(x_valid, y_valid)
    test_dataset = IMDBDataset(x_test, y_test)

    # 데이터 모듈 생성 및 데이터 준비
    imdb_data_module = IMDBDataModule(batch_size=configs.get('batch_size'))
    imdb_data_module.prepare(train_dataset, valid_dataset, test_dataset)

    # 모델 생성
    configs.update({
        'input_dim': x_train.shape[-1],
        'seq_len': x_train.shape[1],
        })
    model = Model(configs)

    # LightningModule 인스턴스 생성
    imdb_module = IMDBModule(
        model=model,
        learning_rate=configs.get('learning_rate'),
    )

    # Trainer 인스턴스 생성 및 설정
    del configs['output_dim'], configs['seed'], configs['epochs'], configs['seq_len'], configs['input_dim']
    exp_name = f'config_{hash(str(configs))}'
    trainer_args = {
        'max_epochs': configs.get('epochs'),
        'callbacks': [
            EarlyStopping(monitor='loss/val_loss', mode='min', patience=10)
        ],
        'logger': TensorBoardLogger(
            'tensorboard',
            f'fine_dust/{exp_name}',
        ),
    }

    if configs.get('device') == 'gpu':
        trainer_args.update({'accelerator': configs.get('device')})

    trainer = Trainer(**trainer_args)

    # 모델 학습 시작
    trainer.fit(
        model=imdb_module,
        datamodule=imdb_data_module,
    )


if __name__ == '__main__':
    # 사용 가능한 GPU가 있는 경우 'cuda', 그렇지 않으면 'cpu' 사용
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    # hyperparameter
    with open('./configs.json', 'r') as file:
        configs = json.load(file)
    configs.update({'device': device})

    # seed 설정s
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
