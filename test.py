'''
test project for data science involving numpy, pandas, matplotlib

'''
'''numpy test'''
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
sum=0
print("Array:",arr)
for i in arr:
	sum+=i
print("Sum:",sum)

'''pandas test'''
import pandas as pd

# Create a simple DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'Salary': [50000, 60000, 70000, 80000]
}
df = pd.DataFrame(data)

# Display the DataFrame
print("Original DataFrame:")
print(df)

# Test 1: Select rows where Age > 30
print("\nTest 1 - Age > 30:")
print(df[df['Age'] > 30])

# Test 2: Calculate the average salary
average_salary = df['Salary'].mean()
print(f"\nTest 2 - Average Salary: {average_salary}")

# Test 3: Add a new column for Salary in thousands
df['Salary_k'] = df['Salary'] / 1000
print("\nTest 3 - DataFrame with Salary_k column:")
print(df)

# Test 4: Sort by Age descending
print("\nTest 4 - Sorted by Age descending:")
print(df.sort_values(by='Age', ascending=False))

'''matplotlib test'''
import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [10, 12, 8, 14, 7]

# Create the plot
plt.plot(x, y, marker='o', linestyle='-', color='blue', label='Sample Line')

# Add labels and title
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Simple Matplotlib Plot')
plt.legend()

# Show the plot
plt.show()