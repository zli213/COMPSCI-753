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


filename = 'bitvector_all_1gram.csv'
num_articles = count_articles(filename)
num_features = count_features(filename)

print(f"Number of articles: {num_articles}")
print(f"Number of features: {num_features}")
