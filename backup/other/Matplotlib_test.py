'''
test project for data science using matplotlib

'''
'''numpy test'''
import matplotlib.pyplot as plt

# Simple matplotlib test: Line plot
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 9, 4, 20, 16]

plt.figure(figsize=(6,4))
plt.plot(x, y, marker='o', linestyle='-', color='b', label='y = x^2')
plt.title("Simple Line Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()