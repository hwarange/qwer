import torch.nn as nn


class Model(nn.Module):  # nn.Module을 상속받아 새로운 모델 클래스를 정의합니다.
    def __init__(self, input_dim, hidden_dim, output_dim):  # 생성자 메서드
        super().__init__()  # 부모 클래스의 생성자를 호출하여 초기화합니다.
        self.input_dim = input_dim  # 입력 차원 크기를 저장합니다.
        self.hidden_dim = hidden_dim  # 숨겨진 층의 차원 크기를 저장합니다.
        self.output_dim = output_dim  # 출력 차원 크기를 저장합니다.

        self.linear1 = nn.Linear(input_dim, hidden_dim)  # 입력 차원에서 숨겨진 차원으로의 선형 변환을 정의합니다.
        self.relu1 = nn.ReLU()  # ReLU 활성화 함수를 정의합니다.
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)  # 입력 차원에서 숨겨진 차원으로의 선형 변환을 정의합니다.
        self.relu2 = nn.ReLU()  # ReLU 활성화 함수를 정의합니다.
        self.output = nn.Linear(hidden_dim, output_dim)  # 숨겨진 차원에서 출력 차원으로의 선형 변환을 정의합니다.
    
    def forward(self, x):  # 순전파 메서드
        x = self.linear1(x)  # 입력 데이터에 대해 선형 변환을 적용합니다.
        x = self.relu1(x)  # ReLU 활성화 함수를 적용하여 비선형성을 추가합니다.
        x = self.linear2(x)  # 입력 데이터에 대해 선형 변환을 적용합니다.
        x = self.relu2(x)  # ReLU 활성화 함수를 적용하여 비선형성을 추가합니다.
        x = self.output(x)  # 두 번째 선형 변환을 적용하여 최종 출력을 계산합니다.

        return x  # 최종 출력을 반환합니다.
