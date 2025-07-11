'''
test project for data science involving numpy, pandas

'''
'''numpy test'''
print("Numpy Test")
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
sum=0
print("Array:",arr)
for i in arr:
	sum+=i
print("Sum:",sum)

'''pandas test'''
import pandas as pd
print("\nPandas Test")
# Create a simple DataFrame
data = {
    'Name': ['Arjun', 'Akshay', 'Raihan', 'Abhishek'],
    'Age': [19, 19, 19, 19],
    'Salary': [50000, 60000, 70000, 80000]
}
df = pd.DataFrame(data)

# Display the DataFrame
print("Original DataFrame:")
print(df)

# Test 1: Select rows where Age > 16
print("\nTest 1 - Age > 16:")
print(df[df['Age'] > 16])

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