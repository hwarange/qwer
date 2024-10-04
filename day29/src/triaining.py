import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L

    
class DiamondsModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,    # 구축한 모델
        learning_rate: float,      # 학습률
    ):
        super().__init__()
        self.model = model         # 모델 초기화
        self.learning_rate = learning_rate  # 학습률 초기화

    def training_step(self, batch, batch_idx):
        X = batch.get('X').to # 입력 데이터를 장치로 이동
        y = batch.get('y').to  # 레이블 데이터를 장치로 이동

        output = self.model(X)  # 모델의 예측값 계산
        self.loss = F.mse_loss(output, y)  # 손실 계산

        return self.loss  # 손실 반환
    
    def on_train_epoch_end(self, *args, **kwargs):
        # 학습 에포크가 끝날 때 호출되는 메서드
        self.log_dict(
            {'loss/train_loss': self.loss},
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
    
    def validation_step(self, batch, batch_idx):
        X = batch.get('X')  # 입력 데이터를 장치로 이동
        y = batch.get('y')  # 레이블 데이터를 장치로 이동

        output = self.model(X)  # 모델의 예측값 계산
        self.val_loss = F.mse_loss(output, y)  # 검증 손실 계산

        return self.val_loss  # 검증 손실 반환
    
    def on_validation_epoch_end(self):
        # 검증 에포크가 끝날 때 호출되는 메서드
        self.log_dict(
            {'loss/val_loss': self.val_loss,
             'learning_rate': self.learning_rate},
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def test_step(self, batch, batch_idx):
        # 테스트 단계에서 호출되는 메서드
        X = batch.get('X')  # 입력 데이터를 장치로 이동
        y = batch.get('y') # 레이블 데이터를 장치로 이동
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
            mode='min',
            factor=0.5,
            patience=5,
        )

        return {
            'optimizer': optimizer,# 옵티마이저 반환
            'scheduler': scheduler,
        }




