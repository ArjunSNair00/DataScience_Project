import pandas as pd
import os

base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "students.csv")
df = pd.read_csv(csv_path)

print(df)
