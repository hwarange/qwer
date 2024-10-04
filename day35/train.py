from src.data import DiamondsDataset, DiamondsDataModule
from src.utils import convert_category_into_integer
from src.model.mlp import Model
from src.training import DiamondsModule

import numpy as np
import random
import json
import nni

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch

import lightning as L
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

import seaborn as sns


def main(configs):
    # 'diamonds' 데이터셋을 로드
    diamonds = sns.load_dataset('diamonds')

    # 결측값이 있는 모든 행 제거
    diamonds = diamonds.dropna()

    # 범주형 열을 정수형으로 변환
    diamonds, _ = convert_category_into_integer(diamonds, ('cut', 'color', 'clarity'))

    # 데이터프레임을 float32로 변환
    diamonds = diamonds.astype(np.float32)

    # 데이터셋을 학습용과 임시 데이터로 분할
    train, temp = train_test_split(diamonds, test_size=0.4, random_state=seed)

    # 임시 데이터를 검증용과 테스트용 데이터로 분할
    valid, test = train_test_split(temp, test_size=0.5, random_state=seed)

    standard_scaler = StandardScaler()

    # 훈련 데이터의 열을 표준화
    train.loc[:, ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']] = \
        standard_scaler.fit_transform(train.loc[:, ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']])

    # 검증 데이터와 테스트 데이터의 열을 훈련 데이터의 통계로 표준화
    valid.loc[:, ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']] = \
        standard_scaler.transform(valid.loc[:, ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']])

    test.loc[:, ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']] = \
        standard_scaler.transform(test.loc[:, ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']])

    # 데이터셋 객체로 변환
    train_dataset = DiamondsDataset(train)
    valid_dataset = DiamondsDataset(valid)
    test_dataset = DiamondsDataset(test)

    # 데이터 모듈 생성 및 데이터 준비
    diamonds_data_module = DiamondsDataModule(batch_size=configs.get('batch_size'))
    diamonds_data_module.prepare(train_dataset, valid_dataset, test_dataset)

    # 모델 생성
    model = Model(len(diamonds.columns)-1, configs.get('hidden_dim'), configs.get('output_dim'))

    # LightningModule 인스턴스 생성
    diamonds_module = DiamondsModule(
        model=model,
        configs= configs,
    )

    # Trainer 인스턴스 생성 및 설정
    del configs['output_dim'], configs['seed']
    exp_name = ','.join([f'{key}={value}' for key, value in configs.items()])
    trainer_args = {
        'max_epochs': configs.get('epochs'),
        'callbacks': [
            EarlyStopping(monitor='loss/val_loss', mode='min', patience=3)
        ],
        'logger': TensorBoardLogger(
            'tensorboard',
            f'diamonds/{exp_name}',
        ),
    }

    if configs.get('device') == 'gpu':
        trainer_args.update({'accelerator': configs.get('device')})

    trainer = Trainer(**trainer_args)

    # 모델 학습 시작
    trainer.fit(
        model=diamonds_module,
        datamodule=diamonds_data_module,
    )

    trainer.test(
        model=diamonds_module,
        datamodule=diamonds_data_module,
    )


if __name__ == '__main__':
    # 사용 가능한 GPU가 있는 경우 'cuda', 그렇지 않으면 'cpu' 사용
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    # hyperparameter
    with open('./configs.json', 'r') as file:
        configs = json.load(file)
    configs.update({'device': device})

    if configs.get('nni'):
        nni_params = nni.get_next_parameter()
        configs.update({'batch_size': nni_params.get('batch_size')})
        configs.update({'hidden_dim': nni_params.get('hidden_dim')})
        configs.update({'learning_rate': nni_params.get('learning_rate')})
        configs.update({'dropout_ratio': nni_params.get('dropout_ratio')})


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