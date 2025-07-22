"""
메모리 효과를 고려한 Lorenz 96 모델 구현
- LSTM 기반 메모리 네트워크
- 과거 10개 시점을 활용한 동역학 예측
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from numba import jit, njit
from functools import partial
from tqdm import tqdm


# =============================================================================
# LSTM 메모리 네트워크 클래스
# =============================================================================
class LSTMMemoryNetwork(torch.nn.Module):
    """
    향상된 LSTM 메모리 네트워크
        
        Args:
        input_dim (int): 입력 차원 (K)
        hidden_dim (int): LSTM 은닉 차원
        output_dim (int): 출력 차원 (K)
        num_layers (int): LSTM 층 수
        dropout (float): 드롭아웃 비율
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=8, num_layers=2, dropout=0.3):
        super(LSTMMemoryNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # BatchNorm을 LSTM 이후로 이동
        self.feature_bn = torch.nn.BatchNorm1d(hidden_dim)
        
        # 입력 전처리 층
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        self.lstm = torch.nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        """
        순전파
        
        Args:
            x (torch.Tensor): 입력 시퀀스 [batch_size, seq_len, input_dim]
            이미 forward_pass_2에서 정규화된 입력
        Returns:
            torch.Tensor: 출력 [batch_size, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 입력 전처리
        x = x.view(-1, self.input_dim)
        x = self.input_layer(x)
        x = x.view(batch_size, seq_len, self.hidden_dim)
        
        # LSTM 처리
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        # LSTM 출력에 대한 BatchNorm
        last_output = self.feature_bn(last_output)
        
        # 출력 생성
        output = self.output_layer(last_output)
        
        return output



if __name__ == "__main__":
    print("메모리 효과를 고려한 Lorenz 96 모델")
    data_list = []
    for i in range(1, 301):
        X_data = np.load(os.path.join(os.getcwd(), "simulated_data", f"X_batch_{i}.npy"))
        Y_data = np.load(os.path.join(os.getcwd(), "simulated_data", f"Y_batch_{i}.npy"))
        C_data = np.load(os.path.join(os.getcwd(), "simulated_data", f"C_batch_{i}.npy"))
        data_list.append([X_data, Y_data, C_data])

