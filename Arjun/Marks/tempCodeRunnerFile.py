import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

directory = os.path.dirname(__file__)
dataset = os.path.join(directory, "students.csv")
students = pd.read_csv(dataset)

marks = students['Marks'].values
results = np.where(marks >= 60, 'Pass', 'Fail')
students['Result'] = results

print("With Pass/Fail Classification:")
print(students)
print("\nInsights:")
print("Mean:", np.mean(marks))
print("Median:", np.median(marks))
print("Max:", np.max(marks))
print("Min:", np.min(marks))

sorted_students = students.sort_values(by='Marks')
print("\nSorted by Marks:")
print(sorted_students)


#matplotlib

#1: Bar chart of Marks
plt.figure(figsize=(10, 5))
plt.bar(students['Name'], students['Marks'], color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.title("Student Marks")
plt.xlabel("Student")
plt.ylabel("Marks")
plt.tight_layout()
plt.savefig("marks_bar_chart.png")
plt.show()

#2: Pie chart of Pass vs Fail
pass_fail_counts = students['Result'].value_counts()

plt.figure(figsize=(5, 5))
plt.pie(pass_fail_counts, labels=pass_fail_counts.index, autopct='%1.1f%%', colors=['green', 'red'])
plt.title("Pass vs Fail Distribution")
plt.savefig("pass_fail_pie_chart.png")
plt.show()