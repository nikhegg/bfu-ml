import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.patches as patches

with open("student_scores.csv", "r") as file:
    data = pd.read_csv(file)

print("Доступные столбцы:")
for i, col in enumerate(data.columns):
    print(f"\t{i}: {col}")

x_col = int(input("Введите столбец для X: "))
y_col = int(input("Введите столбец для Y: "))

x = data.iloc[:, x_col]
y = data.iloc[:, y_col]
    
print("\n-----Статистическая информация-----")
print(f"Количество данных: {len(data)}")
print(f"Минимум:\n\tпо X: {x.min()}\n\tпо Y: {y.min()}")
print(f"Максимум:\n\tпо X: {x.max()}\n\tпо Y: {y.max()}")
print(f"Среднее\n\tпо X: {x.mean()}\n\tпо Y: {y.mean()}")

def linear_regression(x, y):
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x**2)

    n = len(x)
    a = (sum_x * sum_y / n - sum_xy) / (sum_x**2 / n - sum_x2)
    b = (sum_y - a * sum_x) / n
    
    return a, b

a, b = linear_regression(x, y)
print(f"Параметры регрессионной прямой:\n\ta={a}\n\tb={b}")

_, axes = plt.subplots()

axes.scatter(x, y, color='blue')
axes.plot(x, a * x + b, color='red')

for i in range(len(x)):
    y_pred = a * x[i] + b
    err = y[i] - y_pred

    rect_x = x[i]
    rect_y = min(y[i], y_pred)
    size = abs(err)

    rect = patches.Rectangle((rect_x, rect_y), size, size, linewidth=1, edgecolor="green", facecolor="green",alpha=0.25,hatch="//")
    axes.add_patch(rect)

axes.set_xlabel('X')
axes.set_ylabel('Y')
plt.show()