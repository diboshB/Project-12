# train_model.py

import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, mean_squared_log_error
import joblib
import matplotlib.pyplot as plt

# Connecting to the database
conn = sqlite3.connect('/Users/diboshbaruah/Desktop/Database.db')
data = pd.read_sql_query('SELECT * FROM Automobile_data', conn)

print("Dataset successfully loaded...\n")

# Stripping leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Handling Missing Values for Categorical Features
data['Ft'] = data['Ft'].fillna(data['Ft'].mode()[0])
data['Fm'] = data['Fm'].fillna(data['Fm'].mode()[0])

# Removing rows where the target variable 'Fuel consumption' is missing
data = data[data['Fuel consumption'].notnull()]

# Imputing Missing Values for Numerical Features
numerical_columns = ['m (kg)', 'Mt', 'Ewltp (g/km)', 'ec (cm3)', 'ep (KW)', 'z (Wh/km)', 'Erwltp (g/km)']
imputer = SimpleImputer(strategy='median')
existing_numerical_columns = [col for col in numerical_columns if col in data.columns]
data.loc[:, existing_numerical_columns] = imputer.fit_transform(data[existing_numerical_columns])

# Dropping columns with too many missing values 
columns_to_drop = ['z (Wh/km)', 'Erwltp (g/km)']
columns_to_drop = [col for col in columns_to_drop if col in data.columns]
data = data.drop(columns=columns_to_drop)

# Imputing missing Electric range (km) based on fuel type
data['Electric range (km)'] = data.apply(lambda row: 0 if row['Ft'] != 'electric' else row['Electric range (km)'], axis=1)
data['Electric range (km)'] = data['Electric range (km)'].fillna(data['Electric range (km)'].median())

# One-hot encoding categorical columns
data_encoded = pd.get_dummies(data, columns=['Ft', 'Fm'], drop_first=True)

# Splitting the dataset into training, evaluation, and production sets
train_data = data_encoded[:700000]
eval_data = data_encoded[700000:900000]
prod_data = data_encoded[900000:]

# Defining the features (X) and target (y)
X_train = train_data.drop(columns=['Fuel consumption', 'Electric range (km)'])
y_train = train_data['Fuel consumption']

print()
print("Initiating RandomForestRegressor....")
print()

# Initializing and training the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=20, random_state=42)
model.fit(X_train, y_train)

# Saving the trained model to a file
joblib.dump(model, 'random_forest_model.pkl')

# Evaluating model performance
y_pred = model.predict(X_train)
rmse = mean_squared_error(y_train, y_pred) ** 0.5 
mae = mean_absolute_error(y_train, y_pred)
msle = mean_squared_log_error(y_train, y_pred)
evs = explained_variance_score(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

print("Model training complete and saved.")
