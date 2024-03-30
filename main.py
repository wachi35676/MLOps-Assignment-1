# Online Food Analysis

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the data
df = pd.read_csv('onlinefoods.csv')

# Remove the unnamed column
df.drop(columns='Unnamed: 12', inplace=True)

# Overview of Data
print(df.head())  # Print the first few rows
print(df.shape)  # Print the shape (rows, columns)
print(df.describe())  # Print descriptive statistics
print(df.columns)  # Print column names

# Market Segmentation Analysis:

# What is the distribution of individuals across different age groups?
age_counts = df['Age'].value_counts().sort_index()

plt.bar(age_counts.index, age_counts.values, color='green', edgecolor='black')
plt.title('Distribution of Ages', fontsize=16)
plt.xlabel('Ages', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.gcf().patch.set_facecolor('skyblue')
plt.gca().set_facecolor('yellow')
plt.show()

# How does marital status correlate with educational qualifications?
contingency_table = pd.crosstab(
    df['Marital Status'],
    df['Educational Qualifications']
)

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("Chi-square statistic:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)
print("Expected frequencies table:")
print(expected)

# Customer Behavior Analysis:

# How does marital status affect monthly income?
print(pd.DataFrame(df.groupby('Monthly Income'))[0])

# Convert monthly income categories to numeric values
df.loc[df['Monthly Income'] == '10001 to 25000', 'Monthly Income'] = 17500
df.loc[df['Monthly Income'] == '25001 to 50000', 'Monthly Income'] = 37500
df.loc[df['Monthly Income'] == 'Below Rs.10000', 'Monthly Income'] = 10000
df.loc[df['Monthly Income'] == 'More than 50000', 'Monthly Income'] = 50000
df.loc[df['Monthly Income'] == 'No Income', 'Monthly Income'] = 0

df['Monthly Income'] = df['Monthly Income'].astype(int)

print(
    pd.DataFrame(
        df.groupby('Marital Status')['Monthly Income'].mean().sort_values(
            ascending=False
        )
    )
)

# Do students have any significant spending patterns
# or preferences compared to other occupation groups?
income_pattern = df.groupby('Occupation')['Monthly Income'].mean()

plt.figure(figsize=(6, 4))
plt.bar(
    income_pattern.index,
    income_pattern.values,
    color='lightpink',
    edgecolor='black'
)
plt.title('Income Patterns')
plt.xlabel('Occupation')
plt.ylabel('Income')
plt.tight_layout()
plt.xticks(rotation=45)
plt.gcf().patch.set_facecolor('lightgreen')
plt.gca().set_facecolor('lightblue')
plt.show()

# Predictive Analysis
print(df.head())

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns='Monthly Income'),
    df['Monthly Income'],
    test_size=0.2,
    random_state=42
)

print(pd.DataFrame(X_train).head())
print(X_train.shape)

# Define columns for different data types
numerical_columns = [0, 5, 6, 7, 8]
cat_nominal = [1, 2, 3]
cat_ordinal = [5, 9, 10]

# Define pipelines for handling different data types
handle_numerical = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean'))
])

handle_nominal = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(drop='first'))
])

handle_ordinal = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OrdinalEncoder())
])

# Combine the pipelines into a ColumnTransformer
preprocessing = ColumnTransformer(transformers=[
    ('numerical', handle_numerical, numerical_columns),
    ('nominal', handle_nominal, cat_nominal),
    ('ordinal', handle_ordinal, cat_ordinal)
])

# Define the model
model = LinearRegression()

# Create a pipeline for preprocessing and modeling
pipe = make_pipeline(preprocessing, model)

# Fit the pipeline to the training data
pipe.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipe.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2) Score:", r2)
