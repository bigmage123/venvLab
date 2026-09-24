import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

diabetes_X = diabetes_X[:, np.newaxis, 2]

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)

diabetes_y_pred = regr.predict(diabetes_X_test)

print("Коэффициенты:", regr.coef_)
print("Свободный член:", regr.intercept_)
print("Среднеквадратичная ошибка (MSE): %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
print("Коэффициент детерминации (R²): %.2f"
      % r2_score(diabetes_y_test, diabetes_y_pred))

plt.figure(figsize=(10, 6))
plt.scatter(diabetes_X_test, diabetes_y_test, color='steelblue',
            alpha=0.8, label='Тестовые данные')
plt.plot(diabetes_X_test, diabetes_y_pred, color='red',
         linewidth=2, label='Предсказание модели')
plt.xlabel('BMI (нормализованный признак)', fontsize=12)
plt.ylabel('Прогресс диабета (целевая переменная)', fontsize=12)
plt.title('Линейная регрессия на датасете Diabetes\n'
          'R² = %.2f' % r2_score(diabetes_y_test, diabetes_y_pred),
          fontsize=14)
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print("График построен")