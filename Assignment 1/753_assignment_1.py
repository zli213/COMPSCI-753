import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from random import randint
import random
import numpy as np
import pandas as pd
import dask.dataframe as dd


def count_articles(filename):
    df = dd.read_csv(filename, sep='\t', header=None)
    num_articles = df[0].nunique().compute()
    return num_articles


def count_features(filename):
    df = dd.read_csv(filename, sep='\t', header=None)
    # Subtract 1 because one column is the article
    num_features = len(df.columns) - 1
    return num_features


# data set of features
def extract_features(filename):
    # 读取文件，分隔符为制表符
    df = pd.read_csv(filename, sep='\t', header=None)

    # 提取特征列（从第二列到倒数第二列）
    features = df.iloc[:, 1:-1]

    # 将特征转换为整数，并将其作为列表的列表返回
    return features.astype(int).values.T.tolist()


# 示例调用，假设文件名为'bitvector_all_1gram.csv'
filename = 'bitvector_all_1gram.csv'
data = extract_features(filename)

# # 获取data的行数和列数
num_rows = len(data)
num_columns = len(data[0]) if data else 0

# # 打印结果
# print("竖排打印的行数:", num_rows)  # 打印特征数量
# print("一排的列数:", num_columns)  # 打印样本数量

print(f"Number of articles: {num_columns}")
print(f"Number of features: {num_rows}")


# Define the hash function

def hash_function(a, b, n_buckets, x):
    hash_val = (a*x + b)
    return hash_val % n_buckets


def minhash(data, hashfuncs):
    rows, cols, sigrows = num_rows, num_columns, len(hashfuncs)
    sigmatrix = [[float('inf')] * cols for _ in range(sigrows)]

    for r in range(rows):
        hashvalue = [h(r) for h in hashfuncs]
        for c in range(cols):
            if data[r][c] == 0:
                continue
            for i in range(sigrows):
                if sigmatrix[i][c] > hashvalue[i]:
                    sigmatrix[i][c] = hashvalue[i]

    return sigmatrix


def create_hash_functions(k, p):
    hash_funcs = []
    for i in range(k):
        a = randint(1, p-1)
        b = randint(1, p-1)
        hash_funcs.append(lambda x, a=a, b=b: (a * x + b) % p)
    return hash_funcs


# Generate hash functions for k in {2, 4, 8, 16}
ks = [2, 4, 8, 16]
hash_families = {k: create_hash_functions(k, num_rows) for k in ks}

# You can then use these hash functions in your minhash function
for k, hash_funcs in hash_families.items():
    sigmatrix = minhash(data, hash_funcs)
    # print signature matrix
    print(np.array(sigmatrix))
    print(f"Signature matrix shape: {np.array(sigmatrix).shape}")


# 定义参数
m = 600
k = 2

# 假设sigmatrix是你的签名矩阵
rows, cols = len(sigmatrix), len(sigmatrix[0])

# 生成随机哈希函数


def generate_hash_functions(k, m):
    import random
    return [(random.randint(1, m - 1), random.randint(0, m - 1)) for _ in range(k)]


hash_funcs = generate_hash_functions(k, m)

# 构造LSH哈希表


def lsh_hashing(sigmatrix, hash_funcs, m):
    hash_tables = [defaultdict(list) for _ in range(k)]
    for c in range(cols):
        for i, (a, b) in enumerate(hash_funcs):
            bucket_id = (a * sum(sigmatrix[j][c] for j in range(rows)) + b) % m
            hash_tables[i][bucket_id].append(c)
    return hash_tables


hash_tables = lsh_hashing(sigmatrix, hash_funcs, m)

# 报告签名矩阵的维度
print("签名矩阵的行数:", rows)
print("签名矩阵的列数（文章数量）:", cols)

# 选择要分析的哈希表
hash_table = hash_tables[1]

# 计算碰撞分布
collision_distribution = [len(articles) for articles in hash_table.values()]

# 使用matplotlib绘制直方图

plt.hist(collision_distribution, bins=range(
    0, max(collision_distribution) + 1), edgecolor='black')
plt.title('Collision Distribution of Articles into Buckets')
plt.xlabel('Number of Colliding Articles')
plt.ylabel('Frequency')
plt.xticks(range(0, max(collision_distribution) + 1))
plt.show()

# 汇报桶中文章的总和
total_articles = sum(collision_distribution)
print("Total number of articles across buckets:", total_articles)

# 在这里添加你对碰撞分布的分析和评论
