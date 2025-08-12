import os

# OpenMP 중복 라이브러리 로드 문제 해결
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm
import json

# 한글 폰트 설정 제거 (영어로 변경)
plt.rcParams['axes.unicode_minus'] = False

def load_metadata(results_dir):
    """메타데이터 파일을 로드합니다."""
    with open(os.path.join(results_dir, "metadata.json"), "r") as f:
        return json.load(f)

def load_random_batches(results_dir, num_batches=30, total_batches=300):
    """랜덤으로 선택된 배치들을 로드합니다."""
    # 1부터 total_batches까지의 숫자 중에서 num_batches개를 랜덤 선택
    selected_batches = random.sample(range(1, total_batches + 1), num_batches)
    selected_batches.sort()  # 정렬
    
    print(f"Selected batch numbers: {selected_batches}")
    
    all_data = []
    for batch_num in tqdm(selected_batches, desc="Loading batch data"):
        file_path = os.path.join(results_dir, f"X_batch_coupled_{batch_num}.npy")
        if os.path.exists(file_path):
            batch_data = np.load(file_path)
            all_data.append(batch_data)
        else:
            print(f"Warning: {file_path} file does not exist.")
    
    return np.array(all_data), selected_batches

def extract_variable_data(all_batch_data, variable_idx):
    """특정 변수의 데이터를 추출합니다."""
    # all_batch_data shape: (num_batches, batch_size, time_steps, num_variables)
    # 각 배치에서 특정 변수의 시계열 데이터를 추출
    variable_data = []
    
    for batch in all_batch_data:
        # batch shape: (batch_size, time_steps, num_variables)
        # batch_size가 1이므로 batch[0]으로 접근
        time_series = batch[0, :, variable_idx]  # (time_steps,)
        variable_data.append(time_series)
    
    return np.array(variable_data)  # shape: (num_batches, time_steps)

def calculate_correlation_matrix(variable_data):
    """변수 데이터로부터 상관관계 행렬을 계산합니다."""
    # variable_data shape: (num_batches, time_steps)
    # 각 배치 간의 상관관계를 계산
    num_batches = variable_data.shape[0]
    correlation_matrix = np.zeros((num_batches, num_batches))
    
    for i in range(num_batches):
        for j in range(num_batches):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                # 피어슨 상관계수 계산
                corr = np.corrcoef(variable_data[i], variable_data[j])[0, 1]
                correlation_matrix[i, j] = corr
    
    return correlation_matrix

def plot_correlation_matrices(all_batch_data, selected_batches, results_dir, save_dir="correlation_plots"):
    """각 변수별로 상관관계 행렬을 시각화합니다."""
    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    num_variables = all_batch_data.shape[-1]  # 변수 개수 (K=8)
    num_batches = all_batch_data.shape[0]     # 선택된 배치 개수 (30)
    
    print(f"Number of variables: {num_variables}")
    print(f"Number of selected batches: {num_batches}")
    
    # 각 변수별로 상관관계 행렬 계산 및 시각화
    for var_idx in range(num_variables):
        print(f"\nProcessing variable {var_idx + 1}...")
        
        # 변수 데이터 추출
        variable_data = extract_variable_data(all_batch_data, var_idx)
        
        # 상관관계 행렬 계산
        correlation_matrix = calculate_correlation_matrix(variable_data)
        
        # 시각화
        plt.figure(figsize=(12, 10))
        
        # 히트맵 생성
        heatmap = sns.heatmap(correlation_matrix, 
                             annot=True, 
                             cmap='RdBu_r', 
                             center=0,
                             square=True,
                             fmt='.2f',
                             cbar_kws={'label': 'Correlation Coefficient'},
                             xticklabels=selected_batches,
                             yticklabels=selected_batches)
        
        plt.title(f'Variable {var_idx + 1} Correlation Matrix (30x30)', fontsize=16, pad=20)
        plt.xlabel('Batch Number', fontsize=12)
        plt.ylabel('Batch Number', fontsize=12)
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 저장
        save_path = os.path.join(save_dir, f'correlation_matrix_variable_{var_idx + 1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        
        plt.close()
    
    # 모든 변수의 상관관계 행렬을 하나의 그림에 표시
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for var_idx in range(num_variables):
        variable_data = extract_variable_data(all_batch_data, var_idx)
        correlation_matrix = calculate_correlation_matrix(variable_data)
        
        sns.heatmap(correlation_matrix, 
                   ax=axes[var_idx],
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   cbar=False)
        
        axes[var_idx].set_title(f'Variable {var_idx + 1}')
        axes[var_idx].set_xticks([])
        axes[var_idx].set_yticks([])
    
    plt.suptitle('All Variables Correlation Matrices (30x30)', fontsize=16)
    plt.tight_layout()
    
    # 전체 그림 저장
    save_path = os.path.join(save_dir, 'all_variables_correlation_matrices.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"All variables plot saved: {save_path}")
    
    plt.close()

def main():
    # 설정
    results_dir = "simulated_data"
    num_selected_batches = 20
    
    # 메타데이터 로드
    metadata = load_metadata(results_dir)
    total_batches = metadata['num_batches']
    K = metadata['K']
    
    print(f"Total number of batches: {total_batches}")
    print(f"Number of variables (K): {K}")
    print(f"Number of batches to select: {num_selected_batches}")
    
    # 랜덤 시드 설정 (재현 가능성을 위해)
    random.seed(42)
    
    # 랜덤으로 배치 선택 및 데이터 로드
    all_batch_data, selected_batches = load_random_batches(
        results_dir, 
        num_selected_batches, 
        total_batches
    )
    
    print(f"Loaded data shape: {all_batch_data.shape}")
    
    # 상관관계 행렬 시각화
    plot_correlation_matrices(all_batch_data, selected_batches, results_dir)
    
    print("\n=== Analysis Complete ===")
    print("Correlation matrices for each variable have been saved in 'correlation_plots' folder.")

if __name__ == "__main__":
    main() 