import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

1. 
df = pd.read_csv("C:/Users/radhi/Downloads/Housing.csv")
print("Error: Housing.csv not found.")
exit()
print("First 5 rows of the dataset:")
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))
print("\nDataset information:")
print(df.info())
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
df[col] = df[col].map({'yes': 1, 'no': 0})
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True, prefix='furnishing')
print("First 5 rows of the preprocessed dataset:")
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))
print("\nPreprocessed dataset information:")
print(df.info())

2.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
coefficients = pd.DataFrame(lr.coef_, X_train.columns, columns=['Coefficient'])
intercept = lr.intercept_
print("Linear Regression Model Parameters:")
print(f"Intercept: {intercept}")
print("\nCoefficients:")
print(coefficients.to_markdown(numalign="left", stralign="left"))

3.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
plot_df = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Actual Price', y='Predicted Price', data=plot_df)
<Axes: xlabel='Actual Price', ylabel='Predicted Price'>

max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction Line')

plt.title('Actual vs. Predicted House Prices (Linear Regression)')
plt.xlabel('Actual Price (INR)')
plt.ylabel('Predicted Price (INR)')
plt.legend()
plt.grid(True)
plt.show()
coefficients = pd.DataFrame(lr.coef_, X_train.columns, columns=['Coefficient'])
intercept = lr.intercept_
print("\nLinear Regression Model Coefficients:")
print(f"Intercept: {intercept:,.2f}")
print(coefficients.sort_values(by='Coefficient', ascending=False).to_markdown(numalign="left", stralign="left"))

4.
y_pred = lr.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) 
r2 = r2_score(y_test, y_pred)
print("Model Evaluation Metrics (on Test Set):")
print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")

5.
plot_df = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Actual Price', y='Predicted Price', data=plot_df)
max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction line')
plt.title('Actual vs. Predicted House Prices (Linear Regression)')
plt.xlabel('Actual Price (INR)')
plt.ylabel('Predicted Price (INR)')
plt.legend()
plt.grid(True)
plt.show()
coefficients = pd.DataFrame(lr.coef_, X_train.columns, columns=['Coefficient'])
intercept = lr.intercept_
print("\nLinear Regression Model Coefficients:")
print(f"Intercept: {intercept:,.2f}")
print(coefficients.sort_values(by='Coefficient', ascending=False).to_markdown(numalign="left",
stralign="left"))
