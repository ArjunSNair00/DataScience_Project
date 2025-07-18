import csv

# Create a list to store student data
students = []

# Input loop for 10 students
for i in range(1, 11):
    print(f"\nEnter data for Student {i}:")
    name = input("Name: ").strip()
    marks = input("Marks (out of 100): ").strip()

    # Store as a dictionary
    students.append({"Name": name, "Marks": marks})

# Write to CSV
with open("students.csv", mode="w", newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["Name", "Marks"])
    writer.writeheader()  # Write column names
    writer.writerows(students)

print("\n✅ Data saved to 'students.csv'")
