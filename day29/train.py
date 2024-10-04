from src.data import DiamondsDataset, DiamondsDataModule
from src.uitils import convert_category_into_integer
from src.triaining import DiamondsModule
from src.model import Model

import pandas as pd
import numpy as np
import random
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import json

import lightning as L
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

import seaborn as sns



def main(configs):
    # 'diamonds' 데이터셋을 로드
    diamonds = sns.load_dataset('diamonds')

    # 데이터프레임에서 결측값이 있는 모든 행을 제거
    diamonds = diamonds.dropna()

    # 데이터프레임의 범주형 열을 정수형으로 변환
    # 'cut', 'color', 'clarity'
    diamonds, _ = convert_category_into_integer(diamonds, ('cut', 'color', 'clarity'))

    # 데이터프레임의 모든 열을 float32 데이터 타입으로 변환
    diamonds = diamonds.astype(np.float32)

    # 원본 데이터셋을 학습용 데이터와 임시 데이터로 분할
    # 전체 데이터의 40%를 임시 데이터로, 60%를 학습용 데이터로 사용
    train, temp = train_test_split(diamonds, test_size=0.4, random_state=seed)

    # 임시 데이터를 검증용 데이터와 테스트용 데이터로 분할
    # 임시 데이터의 절반을 검증용 데이터로, 나머지 절반을 테스트용 데이터로 사용
    valid, test = train_test_split(temp, test_size=0.5, random_state=seed)


    standard_scaler = StandardScaler()

    # 훈련 세트의 'carat', 'depth', 'table', 'price', 'x', 'y', 'z' 열을 표준화합니다.
    # 표준화는 훈련 데이터의 평균과 표준편차를 사용하여 수행됩니다.
    train.loc[:, ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']] = \
        standard_scaler.fit_transform(train.loc[:, ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']] )

    # 검증 세트의 동일한 열을 훈련 세트에서 계산된 평균과 표준편차를 사용하여 표준화합니다.
    valid.loc[:, ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']] = \
        standard_scaler.transform(valid.loc[:, ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']] )

    # 테스트 세트의 동일한 열을 훈련 세트에서 계산된 평균과 표준편차를 사용하여 표준화합니다.
    test.loc[:, ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']] = \
        standard_scaler.transform(test.loc[:, ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']] )




    # 학습 데이터를 DiamondsDataset 클래스를 사용하여 데이터셋 객체로 변환합니다.
    train_dataset = DiamondsDataset(train)

    # 검증 데이터를 DiamondsDataset 클래스를 사용하여 데이터셋 객체로 변환합니다.
    valid_dataset = DiamondsDataset(valid)

    # 테스트 데이터를 DiamondsDataset 클래스를 사용하여 데이터셋 객체로 변환합니다.
    test_dataset = DiamondsDataset(test)


    diamonds_data_module = DiamondsDataModule(batch_size=configs.get('batch_size'))
    diamonds_data_module.prepare(train_dataset, valid_dataset, test_dataset)

    # len(diamonds.columns) - 1: 입력 특성의 수 (diamonds 데이터셋에서 'survived' 열을 제외한 나머지 열)
    # hidden_dim: 은닉층의 뉴런 수 (변수로 설정된 값)
    # 2: 출력 클래스의 수 (이진 분류 문제이므로 두 개 클래스)
    model = Model(len(diamonds.columns)-1, configs.get('hidden_dim'), 1, configs.get('dropout_prob'),)



    diamonds_module = DiamondsModule(
        model=model,              # 학습할 모델 인스턴스 (예: Model 객체)
        learning_rate=configs.get('learning_rate'),  # 학습률 (예: 1e-3)
        
    )

    # Trainer 인스턴스 생성 및 설정
    trainer_args = {
        'max_epochs': configs.get('epochs'),
        'callbacks': [
            EarlyStopping(monitor='loss/val_loss', mode='min', patience=3)
        ],
        'logger': TensorBoardLogger(
            'tensorboard',
            f'diamonds/seed={configs.get("seed")},batch_size={configs.get("batch_size")},learning_rate={configs.get("learning_rate")},hidden_dim={configs.get("hidden_dim")},dropout_prob={configs.get("dropout_prob")}',
        ),
    }

    if configs.get('device') == 'gpu':
        trainer_args.update({'accelerator': configs.get('device')})

    trainer = Trainer(**trainer_args)

    trainer.fit(
        model=diamonds_module,       # 학습할 모델 인스턴스
        datamodule=diamonds_data_module,  # 데이터셋과 데이터로더를 제공하는 데이터 모듈 인스턴스
    )



if __name__ == '__main__':
    # 사용 가능한 GPU가 있는 경우 'cuda'를, 그렇지 않으면 'cpu'를 사용하도록 장치를 설정합니다.
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    # 시드를 설정하여 실험의 재현성을 '보장합니다.
    with open('./configs.json', 'r') as file:
        configs = json.load(file)
    configs.update({'device':device})


    seed = configs.get('seed')
    random.seed(seed)
    np.random.seed(seed)  # NumPy의 시드를 설정합니다.
    torch.manual_seed(seed)  # PyTorch의 시드를 설정합니다.

    # 장치가 CUDA(GPU)일 경우, CUDA의 시드를 설정합니다.
    if device == 'gpu':
        torch.cuda.manual_seed(seed)  # 현재 장치의 CUDA 시드를 설정합니다.
        torch.cuda.manual_seed_all(seed)  # 모든 CUDA 장치의 시드를 설정합니다.
        torch.backends.cudnn.deterministic = False  # Deterministic 연산을 비활성화합니다. 비활성화하면 더 빠른 연산이 가능할 수 있습니다.
        torch.backends.cudnn.benchmark = True  # 벤치마크를 활성화하여 최적의 성능을 위해 CUDA 커널을 자동으로 튜닝합니다.
    
    main(configs)