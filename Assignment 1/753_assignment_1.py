import time
import pandas as pd
from collections import defaultdict
from random import randint
import numpy as np
import matplotlib.pyplot as plt

# A. Load data and construct feature vectors
filename = 'bitvector_all_1gram.csv'
df = pd.read_csv(filename, sep='\t', header=None)
features = df.iloc[:, 1:-1].astype(int).values.T.tolist()
num_articles = len(features[0]) if features else 0
num_features = len(features)
print(f"Number of articles: {num_articles}")
print(f"Number of features: {num_features}")

# B. Construct a family of MinHash functions


def is_prime(num):
    """Check if a number is prime."""
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True


def create_hash_functions(k, n):
    # Find a prime number p that is greater than n
    p = n + 1
    while not is_prime(p):
        p += 1
    hash_funcs = []
    for _ in range(k):
        a = randint(0, p-1)
        b = randint(0, p-1)
        def func(x, a=a, b=b): return ((a * x + b) % p) % n
        hash_funcs.append((func, a, b))
    return hash_funcs


ks = [2, 4, 8, 16]
hash_families = {k: create_hash_functions(k, num_features) for k in ks}

for col in range(len(features[0])):  # 遍历每一列（即每个文档）
    for row in range(len(features)):  # 遍历每一行（即每个特征或shingle）
        if features[row][col] == 1:
            for k, hash_funcs in hash_families.items():
                print(f"MinHash functions for k={k}, document={col}:")
                for func, a, b in hash_funcs:
                    h_value = func(row)
                    print(
                        f"Function with a={a}, b={b} gives h={h_value} for x={row}")

# C. Construct LSH hash tables


def minhash(data, hashfuncs):
    rows, cols = num_features, num_articles
    sigmatrix = [[float('inf')] * cols for _ in range(len(hashfuncs))]
    for c in range(cols):
        for r in range(rows):
            if data[r][c] == 0:
                continue
            for i, (h, a, b) in enumerate(hashfuncs):
                hash_val = h(r)
                if sigmatrix[i][c] > hash_val:
                    sigmatrix[i][c] = hash_val
    return sigmatrix


def create_level_2_hash_functions(k, num_features, m):
    p = num_features + 1
    while not is_prime(p):
        p += 1
    hash_funcs = []
    for _ in range(k):
        # For c_{i,0}, it can be between 0 and p-1
        c_0 = randint(0, p-1)
        # For other coefficients, they should be between 1 and p-1
        coefficients = [randint(1, p-1) for _ in range(k)]
        coefficients.insert(0, c_0)  # Inserting c_{i,0} at the beginning

        def hash_func(x, coefficients=coefficients):
            assert len(x) == len(
                coefficients) - 1, f"Length mismatch: x has {len(x)} elements, coefficients has {len(coefficients)} elements"
            return (sum(coefficients[i+1] * x[i] for i in range(len(x))) + coefficients[0]) % p % m
        hash_funcs.append((hash_func, coefficients))
    return hash_funcs


def lsh_hashing(sigmatrix, m, k):

    hash_functions = create_level_2_hash_functions(k, num_features, m)

    hash_tables = [defaultdict(list) for _ in range(k)]
    for c in range(len(sigmatrix[0])):  # 遍历每一列（即每个文档）
        column_data = [sigmatrix[j][c]
                       for j in range(len(sigmatrix))]  # 获取整列数据
        for i, (hash_func, _) in enumerate(hash_functions):
            # 使用完整的列数据计算bucket_id
            bucket_id = hash_func(column_data)
            hash_tables[i][bucket_id].append(c)
    return hash_tables


m = 600
k = 2
hash_funcs = hash_families[k]
sigmatrix = minhash(features, hash_funcs)
hash_tables = lsh_hashing(sigmatrix, m, k)
print(
    f"Signature matrix shape: {len(sigmatrix)} rows, {len(sigmatrix[0])} columns")

# D. Compute collision distribution
collision_distribution = [len(articles)
                          for table in hash_tables for articles in table.values()]
plt.hist(collision_distribution, bins=range(
    1, max(collision_distribution) + 1), edgecolor='black')
plt.title('Collision Distribution of Articles into Buckets')
plt.xlabel('Number of Colliding Articles')
plt.ylabel('Frequency')
plt.show()

print("Total number of articles across buckets:", sum(collision_distribution))

# 2. Nearest neighbor search


def jaccard_similarity(set1, set2):
    return len(set1.intersection(set2)) / len(set1.union(set2))


Q = [4996, 4997, 4998, 4999, 5000]
movie_genre = pd.Series(df.iloc[:, -1].values, index=df.iloc[:, 0]).to_dict()

# Pre-compute feature sets for all articles
feature_sets = [set([i for i, x in enumerate(article) if x == 1])
                for article in zip(*features)]

# A. Estimated Jaccard similarity
for q in Q:
    Dq = set()
    for table in hash_tables:
        for bucket in table.values():
            if q in bucket:
                Dq.update(bucket)
    similarities = [
        (d + 1, jaccard_similarity(feature_sets[q-1], feature_sets[d-1])) for d in Dq]
    similarities.sort(key=lambda x: x[1], reverse=True)
    print(
        f"Top 5 articles for query {q} based on estimated Jaccard similarity:")
    for movie_id, sim in similarities[:5]:
        print(f"{movie_id}\t{sim}\t{movie_genre[movie_id]}")

# B. True Jaccard similarity
for q in Q:
    similarities = [(d + 1, jaccard_similarity(feature_sets[q-1], feature_sets[d-1]))
                    for d in range(num_articles)]
    similarities.sort(key=lambda x: x[1], reverse=True)
    print(f"Top 5 articles for query {q} based on true Jaccard similarity:")
    for movie_id, sim in similarities[:5]:
        print(f"{movie_id}\t{sim}\t{movie_genre[movie_id]}")


# Compute MAE for different values of k
Q = list(range(4000, 5001))
maes = []
for k, hash_funcs in hash_families.items():
    sigmatrix = minhash(features, hash_funcs)
    hash_tables = lsh_hashing(sigmatrix, m, k)
    total_error = 0
    for q in Q:
        Dq = set()
        for table in hash_tables:
            for bucket in table.values():
                if q in bucket:
                    Dq.update(bucket)
        for d in Dq:
            estimated_similarity = jaccard_similarity(
                feature_sets[q], feature_sets[d])
            true_similarity = jaccard_similarity(
                feature_sets[q], feature_sets[d])
            total_error += abs(true_similarity - estimated_similarity)
    mae = total_error / (num_articles * len(Q))
    maes.append(mae)

plt.plot(ks, maes)
plt.xlabel('k')
plt.ylabel('MAE')
plt.show()


# B. Compare query times
Q = list(range(4000, 5001))
k = 2
hash_funcs = hash_families[k]
sigmatrix = minhash(features, hash_funcs)
hash_tables = lsh_hashing(sigmatrix, m, k)
print("in 3(B)")
# Question 2(A)
start_time = time.time()
for q in Q:
    Dq = set()
    for table in hash_tables:
        for bucket in table.values():
            if q in bucket:
                Dq.update(bucket)
    similarities = [
        (d + 1, jaccard_similarity(feature_sets[q-1], feature_sets[d-1])) for d in Dq]
    similarities.sort(key=lambda x: x[1], reverse=True)
    print("in 2(A)")
end_time = time.time()
print(
    f"Average query time for Question 2(A): {(end_time - start_time) / len(Q)} ms")

# Question 2(B)
start_time = time.time()
for q in Q:
    similarities = [(d + 1, jaccard_similarity(feature_sets[q-1], feature_sets[d-1]))
                    for d in range(num_articles)]
    similarities.sort(key=lambda x: x[1], reverse=True)
end_time = time.time()
print(
    f"Average query time for Question 2(B): {(end_time - start_time) / len(Q)} ms")
