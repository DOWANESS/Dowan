import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文显示和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data_file = "Data4Regression.xlsx"
train_data = pd.read_excel(data_file, sheet_name=0)
test_data = pd.read_excel(data_file, sheet_name=1)

# 提取训练和测试数据
x_train = train_data.iloc[:, 0].values
y_train = train_data.iloc[:, 1].values
x_test = test_data.iloc[:, 0].values
y_test = test_data.iloc[:, 1].values

# 构建设计矩阵（添加截距项）
X_train = np.column_stack((np.ones(len(x_train)), x_train))
X_test = np.column_stack((np.ones(len(x_test)), x_test))

# 定义均方误差函数
def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 最小二乘法
def least_squares(X, y):
    return np.linalg.inv(X.T @ X) @ (X.T @ y)

theta_ls = least_squares(X_train, y_train)
y_train_pred_ls = X_train @ theta_ls
y_test_pred_ls = X_test @ theta_ls
mse_train_ls = calculate_mse(y_train, y_train_pred_ls)
mse_test_ls = calculate_mse(y_test, y_test_pred_ls)

print("[最小二乘法]")
print(f"训练集 MSE: {mse_train_ls}")
print(f"测试集 MSE: {mse_test_ls}")

# 梯度下降法
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)
    mse_history = []
    for _ in range(iterations):
        gradient = (1 / m) * X.T @ (X @ theta - y)
        theta -= learning_rate * gradient
        mse_history.append(calculate_mse(y, X @ theta))
    return theta, mse_history

theta_gd, mse_history_gd = gradient_descent(X_train, y_train)
y_train_pred_gd = X_train @ theta_gd
y_test_pred_gd = X_test @ theta_gd
mse_train_gd = calculate_mse(y_train, y_train_pred_gd)
mse_test_gd = calculate_mse(y_test, y_test_pred_gd)

print("\n[梯度下降法]")
print(f"训练集 MSE: {mse_train_gd}")
print(f"测试集 MSE: {mse_test_gd}")
print(f"梯度下降法在 {len(mse_history_gd)} 次迭代后收敛，最终训练集 MSE: {mse_history_gd[-1]:.4f}")

# 牛顿法
def newton_method(X, y, iterations=10):
    m, n = X.shape
    theta = np.zeros(n)
    mse_history = []
    H = (1 / m) * (X.T @ X)  # Hessian 矩阵
    H_inv = np.linalg.inv(H)
    for _ in range(iterations):
        gradient = (1 / m) * (X.T @ (X @ theta - y))
        theta -= H_inv @ gradient
        mse_history.append(calculate_mse(y, X @ theta))
    return theta, mse_history

theta_newton, mse_history_newton = newton_method(X_train, y_train)
y_train_pred_newton = X_train @ theta_newton
y_test_pred_newton = X_test @ theta_newton
mse_train_newton = calculate_mse(y_train, y_train_pred_newton)
mse_test_newton = calculate_mse(y_test, y_test_pred_newton)

print("\n[牛顿法]")
print(f"训练集 MSE: {mse_train_newton}")
print(f"测试集 MSE: {mse_test_newton}")
print(f"牛顿法在 {len(mse_history_newton)} 次迭代后收敛，最终训练集 MSE: {mse_history_newton[-1]:.4f}")

# 绘制回归曲线和数据点
plt.figure(figsize=(8, 6))
plt.scatter(x_train, y_train, color='blue', label='训练数据')
plt.scatter(x_test, y_test, color='green', label='测试数据')

x_range = np.linspace(min(x_train.min(), x_test.min()), max(x_train.max(), x_test.max()), 300)
y_range_ls = theta_ls[0] + theta_ls[1] * x_range
y_range_gd = theta_gd[0] + theta_gd[1] * x_range
y_range_newton = theta_newton[0] + theta_newton[1] * x_range

plt.plot(x_range, y_range_ls, color='red', label='最小二乘法')
plt.plot(x_range, y_range_gd, color='orange', linestyle='--', label='梯度下降法')
plt.plot(x_range, y_range_newton, color='purple', linestyle='-.', label='牛顿法')

plt.xlabel("x")
plt.ylabel("y")
plt.title("回归曲线与数据点")
plt.legend()
plt.show()

# 绘制收敛曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(mse_history_gd, label="梯度下降法")
plt.xlabel("迭代次数")
plt.ylabel("训练集 MSE")
plt.title("梯度下降法收敛曲线")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(mse_history_newton, marker='o', label="牛顿法")
plt.xlabel("迭代次数")
plt.ylabel("训练集 MSE")
plt.title("牛顿法收敛曲线")
plt.legend()

plt.tight_layout()
plt.show()