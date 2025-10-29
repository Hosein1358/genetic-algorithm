import pandas as pd

"""
Loading Dataset
"""

path = 'data/Australian Vehicle Prices.csv'
df = pd.read_csv(path)

df.head()

print("\nSummary of the dataset:")
print(df.info())

"""
Converting datatype to num
"""
df = df.drop('Title', axis=1)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
#df['Price'].isna().sum()

df['Seats'] = df['Seats'].str.replace(' Seats', '', regex=False)
df['Seats'] = pd.to_numeric(df['Seats'], errors='coerce')

df['Doors'] = df['Doors'].str.replace(' Doors', '', regex=False)
df['Doors'] = pd.to_numeric(df['Doors'], errors='coerce')

df['Kilometres'] = pd.to_numeric(df['Kilometres'], errors='coerce')

df['CylindersinEngine'] = df['CylindersinEngine'].str.replace(' cyl', '', regex=False)
df['CylindersinEngine'] = pd.to_numeric(df['CylindersinEngine'], errors='coerce')

df['FuelConsumption'] = df['FuelConsumption'].str.replace(' L / 100 km', '', regex=False)
df['FuelConsumption'] = pd.to_numeric(df['FuelConsumption'], errors='coerce')

# extract the numeric part before the 'L'
df['Engine'] = df['Engine'].str.extract(r'(\d+\.?\d*)\s*[lL]')  # regex to capture numbers like 1.5 or 3.0

# convert to numeric (float)
df['Engine'] = pd.to_numeric(df['Engine'], errors='coerce')

df = df.dropna()

df['Price'] = df['Price'].astype(int)
df['Year'] = df['Year'].astype(int)
df['Seats'] = df['Seats'].astype(int)
df['Doors'] = df['Doors'].astype(int)
df['Kilometres'] = df['Kilometres'].astype(int)
df['CylindersinEngine'] = df['CylindersinEngine'].astype(int)

print(df.info())

"""
cleaning Dataset
"""
# Removing outliers in 'Price' Column
print("Removing outliers in Price")
print(f"min: {df['Price'].min()}")
print(f"max: {df['Price'].max()}")
Q1 = df['Price'].quantile(0.25)
print(f"Q1: {Q1}")
Q3 = df['Price'].quantile(0.75)
print(f"Q3: {Q3}")
IQR = Q3 - Q1

# Define upper and lower bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the dataframe
df = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]


# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Handling Missing Values
# Drop rows with missing values
df = df.dropna()

# Drop duplicate rows
df = df.drop_duplicates()

"""
save cleaning data
"""
path_cleaned_data = 'data/australian_vehicle_prices_cleaned.csv'
df.to_csv(path_cleaned_data, index=False)
print("Clean dataset has been saved")

