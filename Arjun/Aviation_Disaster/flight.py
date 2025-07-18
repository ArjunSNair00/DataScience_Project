import pandas as pd
import matplotlib.pyplot as plt
import os

# Set current script directory
BASE_DIR = os.path.dirname(__file__)

# Load CSV
csv_path = os.path.join(BASE_DIR, "flight.csv")
df = pd.read_csv(csv_path)

# Rename columns for easier access
df.columns = ['index', 'date', 'type', 'reg', 'operator', 'fatalities', 'location', 'damage']

# Convert date
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Clean fatalities
df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce').fillna(0).astype(int)

# -----------------------------
# Q1: Is air travel getting safer?
# -----------------------------
yearly_fatalities = df.groupby(df['date'].dt.year)['fatalities'].sum()
deadliest_year = yearly_fatalities.idxmax()
max_fatalities = yearly_fatalities.max()

plt.figure(figsize=(10, 5))
yearly_fatalities.plot(kind='line', title="Total Fatalities per Year")
plt.xlabel("Year")
plt.ylabel("Fatalities")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "fatalities_per_year.png"))
plt.close()

# -----------------------------
# Q2: Aircrafts with most crashes
# -----------------------------
top_aircraft = df['type'].value_counts().head(10)

plt.figure(figsize=(10, 5))
top_aircraft.plot(kind='barh', color='skyblue')
plt.title("Top 10 Aircraft Types Involved in Crashes")
plt.xlabel("Number of Crashes")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "top_aircraft_types.png"))
plt.close()

# -----------------------------
# Q3: Which year was the deadliest?
# Already computed above
# -----------------------------

# -----------------------------
# Q4: Do larger planes have more fatalities?
# -----------------------------
df['is_large'] = df['type'].str.contains("Boeing|Airbus|McDonnell|Lockheed", case=False, na=False)
large_stats = df.groupby('is_large')['fatalities'].mean()

# -----------------------------
# Q5: Are crashes more common in certain regions?
# -----------------------------
df['country'] = df['location'].str.extract(r',\s*(\w+)$')
top_countries = df['country'].value_counts().head(10)

plt.figure(figsize=(10, 5))
top_countries.plot(kind='bar', color='coral')
plt.title("Top 10 Countries with Most Crashes")
plt.ylabel("Number of Crashes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "top_crash_countries.png"))
plt.close()

# -----------------------------
# Summary Output
# -----------------------------
print()
print(f"🔴 Deadliest Year: {deadliest_year} with {max_fatalities} fatalities.")
print()
print(f"📊 Average Fatalities:")
print(f"   - Large Aircraft: {large_stats[True]:.2f}")
print(f"   - Other Aircraft: {large_stats[False]:.2f}")
print()
print("✅ Analysis Complete. Check the following PNG files for charts:")
print("   - fatalities_per_year.png")
print("   - top_aircraft_types.png")
print("   - top_crash_countries.png")
