import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

directory = os.path.dirname(__file__)
dataset = os.path.join(directory, "flight.csv")
flight = pd.read_csv(dataset)

print(flight)