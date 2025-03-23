import os

import numpy as np
import pandas as pd
import torch


class SubgridNN(torch.nn.Module):
    def __init__(self, input_dim=36, hidden_dim=100, output_dim=36):
        """
        Lorenz 96 시스템의 커플링 항을 근사하는 신경망
        
        Args:
            input_dim (int): 입력 차원 (X 변수의 차원)
            hidden_dim (int): 은닉층의 뉴런 수
            output_dim (int): 출력 차원 (커플링 항의 차원)
        """
        super(SubgridNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = torch.nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        """
        신경망 순전파
        
        Args:
            x (torch.Tensor): 입력 텐서, 형태: [batch_size, input_dim]
            
        Returns:
            torch.Tensor: 예측된 커플링 항, 형태: [batch_size, output_dim]
        """
        x = self.fc1(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        
        return x


if __name__ == "__main__":
    # 파라미터 설정
    m_successive_instances = 5
    batch_size = 100
    num_epoch = 2

    # 데이터 로드
    data_list = []
    for i in range(1, 2):
        X_data = np.load(os.path.join(os.getcwd(), "simulated_data", f"X_batch_{i}.npy"))
        Y_data = np.load(os.path.join(os.getcwd(), "simulated_data", f"Y_batch_{i}.npy"))
        t_data = np.load(os.path.join(os.getcwd(), "simulated_data", f"t_batch_{i}.npy"))
        C_data = np.load(os.path.join(os.getcwd(), "simulated_data", f"C_batch_{i}.npy"))

        data_list.append([X_data, Y_data, t_data, C_data])


    # 데이터를 batch_size 만큼 묶어서 학습
    for n in range(num_epoch):
        # 0부터 300 사이에 있는 100개의 정수를 랜덤으로 추출
        sample_idx = np.random.choice(np.arange(301), size=100, replace=False)
        batch = []
        for idx in sample_idx:
            # X_data, Y_data, t_data, C_data 데이터를 텐서로 변환
            X_data = torch.from_numpy(data_list[idx][0]).float()
            Y_data = torch.from_numpy(data_list[idx][1]).float()
            t_data = torch.from_numpy(data_list[idx][2]).float()
            C_data = torch.from_numpy(data_list[idx][3]).float()

            # random_start_point 설정
            random_start_point = np.random.randint(0, len(t_data) - m_successive_instances)

            # ramdom_start_point 에서 m_successive_instances 만큼의 데이터를 추출
            sliced_X_data = X_data[random_start_point:random_start_point + m_successive_instances]
            sliced_Y_data = Y_data[random_start_point:random_start_point + m_successive_instances]
            sliced_t_data = t_data[random_start_point:random_start_point + m_successive_instances]
            sliced_C_data = C_data[random_start_point:random_start_point + m_successive_instances]

            # batch 데이터 추가
            batch.append([sliced_X_data, sliced_Y_data, sliced_t_data, sliced_C_data])

        # batch 데이터를 텐서로 변환
        batch_X_data = torch.stack([batch[i][0] for i in range(batch_size)])
        batch_Y_data = torch.stack([batch[i][1] for i in range(batch_size)])
        batch_t_data = torch.stack([batch[i][2] for i in range(batch_size)])
        batch_C_data = torch.stack([batch[i][3] for i in range(batch_size)])


