from src.data import TipsDataset, TipsDataModule
from src.utils import convert_category_into_integer
from src.model.mlp import Model
from src.training import TipsModule

import numpy as np
import random
import json
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch

import lightning as L
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

import seaborn as sns


def main(configs):
    # 'tips' 데이터셋을 로드
    tips = sns.load_dataset('tips')

    # 결측값이 있는 모든 행 제거
    tips = tips.dropna()

    # 범주형 열을 정수형으로 변환
    tips, _ = convert_category_into_integer(tips, ('sex', 'smoker',	'day', 'time'))

    # 데이터프레임을 float32로 변환
    tips = tips.astype(np.float32)

    # 데이터셋을 학습용과 임시 데이터로 분할
    train, temp = train_test_split(tips, test_size=0.4, random_state=seed)

    # 임시 데이터를 검증용과 테스트용 데이터로 분할
    valid, test = train_test_split(temp, test_size=0.5, random_state=seed)

    standard_scaler = StandardScaler()

    # 훈련 데이터의 열을 표준화
    train.loc[:, ['total_bill', 'tip']] = \
        standard_scaler.fit_transform(train.loc[:, ['total_bill', 'tip']])

    # 검증 데이터와 테스트 데이터의 열을 훈련 데이터의 통계로 표준화
    valid.loc[:, ['total_bill', 'tip']] = \
        standard_scaler.transform(valid.loc[:, ['total_bill', 'tip']])

    test.loc[:, ['total_bill', 'tip']] = \
        standard_scaler.transform(test.loc[:, ['total_bill', 'tip']])

    # 데이터셋 객체로 변환
    train_dataset = TipsDataset(train)
    valid_dataset = TipsDataset(valid)
    test_dataset = TipsDataset(test)

    # 데이터 모듈 생성 및 데이터 준비
    tips_data_module = TipsDataModule(batch_size=configs.get('batch_size'))
    tips_data_module.prepare(train_dataset, valid_dataset, test_dataset)

    # 모델 생성
    configs.update({'input_dim': len(tips.columns)-1})
    model = Model(configs)

    # LightningModule 인스턴스 생성
    tips_module = TipsModule(
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
            f'tips/{exp_name}',
        ),
    }

    if configs.get('device') == 'gpu':
        trainer_args.update({'accelerator': configs.get('device')})

    trainer = Trainer(**trainer_args)

    # 모델 학습 시작
    trainer.fit(
        model=tips_module,
        datamodule=tips_data_module,
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