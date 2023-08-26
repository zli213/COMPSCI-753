import pandas as pd
import matplotlib.pyplot as plt

# Column names
column_names = ['news_id', 'news_category', 'date']

# Load the dataset using comma as the delimiter and specify column names
df = pd.read_csv('news_stream.csv', delimiter=',',
                 header=None, names=column_names)

# Group by the 'news_category' column and count the occurrences
category_counts = df['news_category'].value_counts()

# 1.A Compute the average frequency
category_frequency = category_counts / len(df)
average_frequency = category_frequency.mean()

print(average_frequency)
# 1.B Compute the true frequencies of all categories
# Sort the frequencies in descending order
sorted_frequency = category_frequency.sort_values(ascending=False)

# Plot the bar chart
plt.figure(figsize=(12, 6))
sorted_frequency.plot(kind='bar')
plt.ylabel('True Frequency')
plt.xlabel('News Category')
plt.title('True Frequencies of News Categories')
plt.tight_layout()
plt.show()
