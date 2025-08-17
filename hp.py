# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# %%
# Load data
data = pd.read_csv('data.csv')

# %%
# Split into features and target
x = data.drop(['price'], axis=1)
y = data['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# %%
# Feature preparation function
def prepare_features(df):
    df = df.copy()
    # Log-transform numeric columns (avoid log(0))
    for col in ['age','amneties','area','atmDistance','balconies','bathrooms',
                'hospitalDistance','restrauntDistance','schoolDistance','shoppingDistance','status']:
        df[col] = np.log1p(df[col])
    
    # Derived features
    df['bathrooms_ratio'] = df['bathrooms'] / df['area']
    df['total_rooms_est'] = df['bathrooms'] + df['balconies'] + 1  # +1 for living room
    df['households_est'] = 1
    df['household_rooms'] = df['total_rooms_est'] / df['households_est']
    
    return df

# %%
# Prepare train and test data
x_train_prepared = prepare_features(x_train)
x_test_prepared = prepare_features(x_test)

# Ensure columns match exactly
x_test_prepared = x_test_prepared[x_train_prepared.columns]

# %%
# Train Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(x_train_prepared, y_train)

# %%
# Predict and evaluate
y_pred = rf.predict(x_test_prepared)
r2 = r2_score(y_test, y_pred)
print(f"R² on test data: {r2}")
