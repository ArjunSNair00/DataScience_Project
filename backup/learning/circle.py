import math

radius = 10

# Adjust the aspect ratio for console characters (they're taller than wide)
for y in range(-radius, radius + 1):
    for x in range(-2 * radius, 2 * radius + 1):
        if abs(math.sqrt((x / 2)**2 + y**2) - radius) < 0.5:
            print("*", end="")
        else:
            print(" ", end="")
    print()
