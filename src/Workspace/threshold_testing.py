import matplotlib.pyplot as plt
import numpy as np

# Example data
list1 = [1, 2, 3, 4, 5]
list2 = [2, 4, 6, 8, 10]

# Calculate mean and standard deviation for each list
mean1 = np.mean(list1)
std1 = np.std(list1)

mean2 = np.mean(list2)
std2 = np.std(list2)

# Plot the histograms
#plt.hist(list1, bins='auto', alpha=0.7, label='List 1')
#plt.hist(list2, bins='auto', alpha=0.7, label='List 2')
plt.axvline(mean1, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean1:.2f}')
plt.axvline(mean2, color='g', linestyle='dashed', linewidth=1, label=f'Mean: {mean2:.2f}')
plt.axvline(mean1+std1, color='r', linestyle='dotted', linewidth=1, label=f'St. Dev.: {std1:.2f}')
plt.axvline(mean1-std1, color='r', linestyle='dotted', linewidth=1)
plt.axvline(mean2+std2, color='g', linestyle='dotted', linewidth=1, label=f'St. Dev.: {std2:.2f}')
plt.axvline(mean2-std2, color='g', linestyle='dotted', linewidth=1)

plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of List 1 and List 2')
plt.legend()

# Display the plot inline in VS Code
plt.show()

