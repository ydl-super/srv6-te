import numpy as np

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 生成一个 132x12 的概率矩阵
probability_matrix = np.random.rand(132, 12)
# 确保每一行的概率之和为1
probability_matrix = probability_matrix / probability_matrix.sum(axis=1, keepdims=True)


# 每一行进行概率抽取
def sample_from_probability_matrix(probability_matrix):
    num_rows, num_cols = probability_matrix.shape
    sampled_indices = np.zeros(num_rows, dtype=int)

    for i in range(num_rows):
        # 使用 np.random.choice 进行概率抽取
        sampled_indices[i] = np.random.choice(num_cols, p=probability_matrix[i])

    return sampled_indices


# 生成 132x1 的策略矩阵
strategy_matrix = sample_from_probability_matrix(probability_matrix)

# 将结果转换为 132x1 的矩阵
strategy_matrix = strategy_matrix.reshape(-1, 1)

# 打印结果
print("Probability Matrix (first 5 rows):")
print(probability_matrix[:5])
print("\nStrategy Matrix (first 5 rows):")
print(strategy_matrix[:5])
