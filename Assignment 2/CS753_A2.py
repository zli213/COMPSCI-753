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

# 2 Misra-Gries Approach and Performance Evaluation


def misra_gries(stream, k):
    counters = {}
    decrement_steps = 0  # Initialize a counter for decrement steps

    for item in stream:
        if item in counters:
            counters[item] += 1
        elif len(counters) < k - 1:
            counters[item] = 1
        else:
            decrement_steps += 1  # Increment the counter when decrementing all items
            for key in list(counters.keys()):
                counters[key] -= 1
                if counters[key] == 0:
                    del counters[key]
    return counters, decrement_steps


# Apply Misra-Gries algorithm
k = 20
estimated_frequencies, num_decrements = misra_gries(df['news_category'], k)

# Sort the estimated frequencies in descending order
sorted_frequencies = dict(
    sorted(estimated_frequencies.items(), key=lambda item: item[1], reverse=True))

# 2.A Plot the estimated frequencies
plt.figure(figsize=(12, 6))
plt.bar(sorted_frequencies.keys(), sorted_frequencies.values())
plt.ylabel('Estimated Frequency')
plt.xlabel('News Category')
plt.title('Misra-Gries Estimated Frequencies of News Categories')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2.B Compare the estimated and true frequencies
# Plot the estimated and true frequencies
plt.figure(figsize=(15, 7))
categories = list(sorted_frequencies.keys())
estimated_values = [sorted_frequencies[cat] for cat in categories]
# Multiply by len(df) to get counts instead of proportions
true_values = [category_frequency.get(cat, 0) * len(df) for cat in categories]

bar_width = 0.35
index = range(len(categories))

bar1 = plt.bar(index, estimated_values, bar_width,
               label='Estimated (Misra-Gries)', color='b', align='center')
bar2 = plt.bar([i + bar_width for i in index], true_values,
               bar_width, label='True Frequencies', color='r', align='center')

plt.xlabel('News Category')
plt.ylabel('Frequency')
plt.title('Comparison of Estimated (Misra-Gries) and True Frequencies')
plt.xticks([i + bar_width / 2 for i in index], categories, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# 2.C Report the actual number of decrement steps
# Print the number of rows in the dataset
print(f"Number of rows in the dataset: {len(df)}")

print(f"Number of decrement steps with k={k}: {num_decrements}")

# 2.D
# Assuming you have a MisraGries class or function implemented.
# pseudo-code for the entire process


def compute_average_absolute_error(k_values, true_frequencies, stream):
    average_errors = []

    for k in k_values:
        # Run Misra-Gries algorithm with the given k
        estimated_counts, num_decrements = misra_gries(stream, k)

        # compute the absolute error for each category and accumulate
        total_error = 0
        for category, true_frequency in true_frequencies.items():
            # Multiply by len(stream) to get counts instead of proportions
            estimated_frequency = estimated_counts.get(category, 0)/len(stream)
            total_error += abs(estimated_frequency - true_frequency)

        average_errors.append(total_error / len(true_frequencies))

    return average_errors


k_values = [10, 20, 30, 40]
# Assuming true_frequencies is a dictionary with categories as keys and true frequencies as values.
# And stream is a list of items representing the data stream.
average_errors = compute_average_absolute_error(
    k_values, category_frequency, df['news_category'])

plt.plot(k_values, average_errors, marker='o')
plt.xlabel("Summary Size k")
plt.ylabel("Average Absolute Error")
plt.title("Impact of Summary Size k on Average Absolute Error")
plt.show()
