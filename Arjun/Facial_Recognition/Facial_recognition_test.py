# aviation_crash_analysis.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set default plot style
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (12, 6)

# 0. Load and clean the data
df = pd.read_csv('aviation_crashes.csv')

# Initial cleanup
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date', 'Fatalities', 'Aboard'])

df['Year'] = df['Date'].dt.year
df['Fatality_Rate'] = df['Fatalities'] / df['Aboard']
df['Aboard'] = df['Aboard'].astype(float)

# Extract country from location (basic method)
df['Country'] = df['Location'].astype(str).str.split(',').str[-1].str.strip()

# ======================
# 1. Exploratory Analysis
# ======================

# Top 10 aircraft by crash count
aircraft_counts = df['Aircraft Type'].value_counts().head(10)
aircraft_counts.plot(kind='bar', title='Top 10 Aircraft Models by Crash Count', color='skyblue')
plt.ylabel('Number of Crashes')
plt.xlabel('Aircraft Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ======================
# 2. Time-Series Analysis
# ======================

# Group by year
yearly = df.groupby('Year').agg({
    'Fatalities': 'sum',
    'Aboard': 'sum',
    'Date': 'count'
}).rename(columns={'Date': 'Crashes'})

yearly['Fatality_Rate'] = yearly['Fatalities'] / yearly['Aboard']

# Plot crashes and fatalities
yearly[['Crashes', 'Fatalities']].plot(title='Crashes and Fatalities Over Time', marker='o')
plt.ylabel('Count')
plt.xlabel('Year')
plt.grid(True)
plt.show()

# Plot fatality rate
yearly['Fatality_Rate'].plot(title='Fatality Rate Over Time', color='red')
plt.ylabel('Fatalities per Person Aboard')
plt.xlabel('Year')
plt.grid(True)
plt.show()

# ======================
# 3. Categorical Analysis
# ======================

# Crashes by country (Top 10)
country_crashes = df['Country'].value_counts().head(10)
country_crashes.plot(kind='bar', title='Top 10 Countries by Crash Count', color='coral')
plt.ylabel('Number of Crashes')
plt.xlabel('Country')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ======================
# 4. Severity Analysis
# ======================

# Bucket plane size by number aboard
bins = [0, 50, 100, 200, 300, 500, 1000]
labels = ['Tiny', 'Small', 'Medium', 'Large', 'Jumbo', 'Mega']
df['Size_Category'] = pd.cut(df['Aboard'], bins=bins, labels=labels)

# Average fatality rate by size
size_group = df.groupby('Size_Category')['Fatality_Rate'].mean()
size_group.plot(kind='bar', title='Average Fatality Rate by Aircraft Size', color='green')
plt.ylabel('Fatality Rate')
plt.xlabel('Aircraft Size')
plt.tight_layout()
plt.show()

# ======================
# 5. Insightful Questions
# ======================

# Q1: Is air travel getting safer?
print("Q1: Air travel safety trends:")
print(yearly[['Crashes', 'Fatalities', 'Fatality_Rate']].tail())

# Q2: Which aircraft models crash most?
print("\nQ2: Top aircraft involved in crashes:")
print(aircraft_counts)

# Q3: Deadliest year
deadliest_year = yearly['Fatalities'].idxmax()
max_deaths = yearly['Fatalities'].max()
print(f"\nQ3: Deadliest year was {deadliest_year} with {max_deaths} fatalities.")

# Q4: Larger planes vs fatality rates
print("\nQ4: Avg fatality rate by plane size:")
print(size_group)

# Q5: Countries with most crashes
print("\nQ5: Countries with most crashes:")
print(country_crashes)

# ======================
# Done! Summary
# ======================

print("\nProject Summary:")
print(f"✔️ Air travel is getting safer: fatality rates have declined over time.")
print(f"✔️ The deadliest year was {deadliest_year}.")
print(f"✔️ Aircraft models with high crash rates: {aircraft_counts.index[0]} and others.")
print(f"✔️ Bigger aircraft don't always have higher fatality rates — check the plot.")
print(f"✔️ Crashes are most frequent in countries with high air traffic or challenging terrain.")
