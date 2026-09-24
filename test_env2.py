import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Backend для отображения окна
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv(
    'bottle.csv',
    usecols=['Salnty', 'T_degC'],
    dtype={'Salnty': 'float64', 'T_degC': 'float64'},
    nrows=1000
)

print("Размер датасета:", df.shape)

df_binary = df.copy()
df_binary.columns = ['Sal', 'Temp']

df_binary = df_binary.ffill()
df_binary.dropna(inplace=True)

print("Размер после очистки:", df_binary.shape)

df_binary500 = df_binary.iloc[:500]

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
r2 = regr.score(X_test, y_test)

print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='steelblue', alpha=0.7,
            label='Тестовые данные')
plt.plot(X_test, y_pred, color='red', linewidth=2,
         label='Предсказание модели')
plt.xlabel('Salinity (Salnty)', fontsize=12)
plt.ylabel('Temperature (T_degC)', fontsize=12)
plt.title(f'Линейная регрессия: Salinity → Temperature\nR² = {r2:.4f}',
          fontsize=14)
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('bottle_regression_plot.png', dpi=100)
plt.show()

print("График сохранён в bottle_regression_plot.png")
