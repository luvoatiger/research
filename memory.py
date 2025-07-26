import os
import sys
import time
import random
import torch

import numpy as np
import pandas as pd


if __name__ == "__main__":
    print("Hello, World!")

        # 데이터 로드
    data_list = []
    for i in range(1, 301):
        X_data = np.load(os.path.join(os.getcwd(), "simulated_data", f"X_batch_{i}.npy"))
        Y_data = np.load(os.path.join(os.getcwd(), "simulated_data", f"Y_batch_{i}.npy"))
        C_data = np.load(os.path.join(os.getcwd(), "simulated_data", f"C_batch_{i}.npy"))
        data_list.append([X_data, Y_data, C_data])
