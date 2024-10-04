import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L


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