import pandas as pd
import numpy as np
import random
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as ds
import torchvision.transforms as transforms

import lightning as L
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def convert_category_into_integer(df: pd.DataFrame, columns: list):
    """
    주어진 DataFrame의 특정 열들을 범주형에서 정수형으로 변환합니다.
    
    Parameters:
    - df (pd.DataFrame): 변환할 데이터프레임
    - columns (list): 범주형에서 정수형으로 변환할 열 이름의 리스트
    
    Returns:
    - pd.DataFrame: 변환된 데이터프레임
    - dict: 각 열에 대해 적합한 LabelEncoder 객체를 포함하는 딕셔너리
    """
    label_encoders = {}  # 각 열의 LabelEncoder 객체를 저장할 딕셔너리입니다.
    
    for column in columns:
        # 각 열에 대해 LabelEncoder 객체를 생성합니다.
        label_encoder = LabelEncoder()
        
        # LabelEncoder를 사용하여 해당 열의 범주형 데이터를 정수형으로 변환합니다.
        df.loc[:, column] = label_encoder.fit_transform(df[column])
        
        # 변환된 LabelEncoder 객체를 딕셔너리에 저장합니다.
        label_encoders.update({column: label_encoder})
    
    # 변환된 데이터프레임과 LabelEncoder 객체를 포함하는 딕셔너리를 반환합니다.
    return df, label_encoders

class DiamondsDataset(Dataset):
    def __init__(self, data):  # 생성자 메서드
        super().__init__()  # 부모 클래스의 생성자를 호출하여 초기화합니다.
        self.data = data  # 데이터프레임을 저장합니다.
    
    def __len__(self):
        return len(self.data)  # 데이터셋의 전체 샘플 수를 반환합니다.
    
    def __getitem__(self, idx):
        # 인덱스 `idx`에 해당하는 데이터 샘플을 반환합니다.
        
        # 데이터프레임에서 'price' 열을 제외한 특성값을 가져와서 NumPy 배열로 변환한 뒤, PyTorch 텐서로 변환합니다.
        X = torch.from_numpy(self.data.iloc[idx].drop('price').values).float()
        
        # 'price' 열의 값을 텐서로 변환하여 레이블을 생성합니다.
        y = torch.Tensor([self.data.iloc[idx].price]).float()
        
        # 입력 데이터와 레이블을 딕셔너리 형태로 반환합니다.
        return {
            'X': X,
            'y': y,
        }
    
class Model(nn.Module):  # nn.Module을 상속받아 새로운 모델 클래스를 정의합니다.
    def __init__(self, input_dim, hidden_dim, output_dim):  # 생성자 메서드
        super().__init__()  # 부모 클래스의 생성자를 호출하여 초기화합니다.
        self.input_dim = input_dim  # 입력 차원 크기를 저장합니다.
        self.hidden_dim = hidden_dim  # 숨겨진 층의 차원 크기를 저장합니다.
        self.output_dim = output_dim  # 출력 차원 크기를 저장합니다.

        self.linear = nn.Linear(input_dim, hidden_dim)  # 입력 차원에서 숨겨진 차원으로의 선형 변환을 정의합니다.
        self.relu = nn.ReLU()  # ReLU 활성화 함수를 정의합니다.
        self.output = nn.Linear(hidden_dim, output_dim)  # 숨겨진 차원에서 출력 차원으로의 선형 변환을 정의합니다.
    
    def forward(self, x):  # 순전파 메서드
        x = self.linear(x)  # 입력 데이터에 대해 선형 변환을 적용합니다.
        x = self.relu(x)  # ReLU 활성화 함수를 적용하여 비선형성을 추가합니다.
        x = self.output(x)  # 두 번째 선형 변환을 적용하여 최종 출력을 계산합니다.

        return x  # 최종 출력을 반환합니다.
    
class DiamondsDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        ):
        super().__init__()
        self.batch_size = batch_size

    def prepare(self, train_dataset, valid_dataset, test_dataset):     
        self.train_dataset, self.valid_dataset, self.test_dataset = train_dataset, valid_dataset, test_dataset

    def setup(self, stage: str):
        if stage == "fit":      
            self.train_data = self.train_dataset
            self.valid_data = self.valid_dataset

        if stage == "test":     
            self.test_data = self.test_dataset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_data,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
        )
    
class DiamondsModule(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,    # 구축한 모델
        learning_rate: float,      # 학습률
    ):
        super().__init__()
        self.model = model         # 모델 초기화
        self.learning_rate = learning_rate  # 학습률 초기화
    
    def training_step(self, batch):

        X = batch.get('X')  # 입력 데이터를 장치로 이동
        y = batch.get('y')  # 레이블 데이터를 장치로 이동
         # 레이블의 차원을 축소

        output = self.model(X)  # 모델의 예측값 계산
        #logit = F.softmax(output, dim=-1)  # 소프트맥스 활성화 함수 적용
        self.loss = F.mse_loss(output, y)  # 손실 계산

        # predicted_label = self.loss.argmax(dim=-1)  # 예측된 레이블 계산
        # self.acc = (predicted_label == y).float().mean()  # 정확도 계산


        return self.loss  # 손실 반환
    
    def on_train_epoch_end(self, *args, **kwargs):
        # 학습 에포크가 끝날 때 호출되는 메서드
        self.log_dict({'loss/train_loss': self.loss}, on_epoch=True, prog_bar=True, logger=True)

       
    
    def validation_step(self, batch):

        X = batch.get('X')  # 입력 데이터를 장치로 이동
        y = batch.get('y')  # 레이블 데이터를 장치로 이동  # 레이블의 차원을 축소

        output = self.model(X)  # 모델의 예측값 계산
        # logit = F.softmax(output, dim=-1)  # 소프트맥스 활성화 함수 적용
        self.val_loss = F.mse_loss(output, y)  # 검증 손실 계산

        predicted_label = self.val_loss.argmax(dim=-1)  # 예측된 레이블 계산
        self.val_acc = (predicted_label == y).float().mean()  # 정확도 계산


        return self.val_loss  # 검증 손실 반환
    
    def on_validation_epoch_end(self):
        # 검증 에포크가 끝날 때 호출되는 메서드
        self.log_dict({'loss/val_loss': self.val_loss}, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict({'learning_rate': self.learning_rate}, on_epoch=True, prog_bar=True, logger=True)
    
    def test_step(self, batch, batch_idx):
        # 테스트 단계에서 호출되는 메서드
        X = batch.get('X').to(device)  # 입력 데이터를 장치로 이동
        y = batch.get('y').to(device)  # 레이블 데이터를 장치로 이동
        y = y.squeeze()  # 레이블의 차원을 축소

        output = self.model(X)  # 모델의 예측값 계산
    

        return output  # 예측된 레이블 반환

    def configure_optimizers(self):
        # 옵티마이저를 설정하는 메서드
        optimizer = optim.Adam(
            self.model.parameters(),  # 모델 파라미터를 옵티마이저에 전달
            lr=self.learning_rate,    # 학습률 설정
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode= 'min',
            factor= 0.5,
            patience= 5,
        )
   
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }  # 옵티마이저 반환
    
    
    

def main():
    diamonds = sns.load_dataset('diamonds')
    diamonds = diamonds.drop_duplicates().reset_index(drop=True)
    diamonds, _ = convert_category_into_integer(diamonds, ('cut', 'color', 'clarity'))

    train, temp = train_test_split(diamonds, test_size=0.4, random_state=seed)

    # 임시 데이터를 검증용 데이터와 테스트용 데이터로 분할
    # 임시 데이터의 절반을 검증용 데이터로, 나머지 절반을 테스트용 데이터로 사용
    valid, test = train_test_split(temp, test_size=0.5, random_state=seed)

    standard_scaler = StandardScaler()

    train.loc[:, ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']] = \
        standard_scaler.fit_transform(train.loc[:, ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']] )

    # 검증 세트의 동일한 열을 훈련 세트에서 계산된 평균과 표준편차를 사용하여 표준화합니다.
    valid.loc[:, ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']] = \
        standard_scaler.transform(valid.loc[:, ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']] )

    # 테스트 세트의 동일한 열을 훈련 세트에서 계산된 평균과 표준편차를 사용하여 표준화합니다.
    test.loc[:, ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']] = \
        standard_scaler.transform(test.loc[:, ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']] )

        # 학습 데이터를 TitanicDataset 클래스를 사용하여 데이터셋 객체로 변환합니다.
    train_dataset = DiamondsDataset(train)

    # 검증 데이터를 TitanicDataset 클래스를 사용하여 데이터셋 객체로 변환합니다.
    valid_dataset = DiamondsDataset(valid)

    # 테스트 데이터를 TitanicDataset 클래스를 사용하여 데이터셋 객체로 변환합니다.
    test_dataset = DiamondsDataset(test)

    model = Model(
        input_dim=len(diamonds.columns)-1,  # 입력 차원 크기를 설정합니다. diamonds 데이터프레임의 열 개수에서 1을 뺀 값입니다.
        hidden_dim=hidden_dim,  # 숨겨진 층의 차원 크기를 설정합니다. 이 값은 미리 정의된 `hidden_dim` 변수로부터 가져옵니다.
        output_dim=1,  # 출력 차원 크기를 설정합니다. 'cut' 열의 고유한 값의 개수를 가져옵니다.
    ).to(device)
    # len(titanic.columns) - 1: 입력 특성의 수 (타이타닉 데이터셋에서 'survived' 열을 제외한 나머지 열)
    # hidden_dim: 은닉층의 뉴런 수 (변수로 설정된 값)
    # 2: 출력 클래스의 수 (이진 분류 문제이므로 두 개 클래스)
    # Model 클래스를 사용하여 모델 인스턴스를 생성합니다.



        
    diamonds_data_module = DiamondsDataModule(batch_size=batch_size)
    diamonds_data_module.prepare(train_dataset, valid_dataset, test_dataset)


    # TitanicModule 클래스의 인스턴스를 생성합니다.
    diamonds_module = DiamondsModule(
        model=model,              # 학습할 모델 인스턴스 (예: Model 객체)
        learning_rate=learning_rate  # 학습률 (예: 1e-3)
    )
    # Trainer 인스턴스 생성
    trainer = Trainer(
        max_epochs=epochs,  # 학습할 최대 에포크 수
        callbacks=[
            EarlyStopping(monitor='loss/val_loss', mode='min', patience= 5)
        ],
        logger= TensorBoardLogger(
            "tensorboard",                                                # root folder
            f"seed= {seed}, batch_size= {batch_size}, learning_rate= {learning_rate}, hidden_dim= {hidden_dim}",   # folder name
        ),
    )
    # PyTorch Lightning Trainer를 사용하여 모델 학습을 시작
    # 'titanic_module'은 학습할 모델과 학습 및 검증 루프를 정의한 LightningModule 인스턴스
    # 'titanic_data_module'은 데이터셋을 로드하고 데이터로더를 제공하는 LightningDataModule 인스턴스
    trainer.fit(
        model=diamonds_module,       # 학습할 모델 인스턴스
        datamodule=diamonds_data_module,  # 데이터셋과 데이터로더를 제공하는 데이터 모듈 인스턴스
    )


if __name__ == '__main__' :
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 시드를 설정하여 실험의 재현성을 보장합니다.
    
    batch_size = 1024  # 데이터 배치의 크기를 설정합니다.
    epochs = 100  # 학습 에포크 수를 설정합니다.
    learning_rate = 1e-1  # 학습률을 설정합니다.
    hidden_dim = 32  # 숨겨진 층의 차원 크기를 설정합니다.
    seed = 0

    random.seed(seed)
    np.random.seed(seed)  # NumPy의 시드를 설정합니다.
    torch.manual_seed(seed)  # PyTorch의 시드를 설정합니다.

    # 장치가 CUDA(GPU)일 경우, CUDA의 시드를 설정합니다.
    if device == 'cuda':
        torch.cuda.manual_seed(seed)  # 현재 장치의 CUDA 시드를 설정합니다.
        torch.cuda.manual_seed_all(seed)  # 모든 CUDA 장치의 시드를 설정합니다.
        torch.backends.cudnn.deterministic = False  # Deterministic 연산을 비활성화합니다. 비활성화하면 더 빠른 연산이 가능할 수 있습니다.
        torch.backends.cudnn.benchmark = True  # 벤치마크를 활성화하여 최적의 성능을 위해 CUDA 커널을 자동으로 튜닝합니다.

    main()
