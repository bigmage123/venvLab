import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('bottle.csv')

print("Размер датасета:", df.shape)

df_binary = df[['Salnty', 'T_degC']].copy()
df_binary.columns = ['Sal', 'Temp']

df_binary = df_binary.ffill()
df_binary.dropna(inplace=True)

print("Размер после очистки:", df_binary.shape)

df_binary500 = df_binary.iloc[:500].copy()

X = np.array(df_binary500['Sal']).reshape(-1, 1)
y = np.array(df_binary500['Temp']).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

regr = LinearRegression()
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
rmse = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)

print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {regr.score(X_test, y_test):.4f}")

plt.scatter(X_test, y_test, color='b', label='Тестовые данные')
plt.plot(X_test, y_pred, color='k', label='Предсказание модели')
plt.xlabel('Salinity (Salnty)')
plt.ylabel('Temperature (T_degC)')
plt.title('Линейная регрессия: Salinity → Temperature')
plt.legend()
plt.show()