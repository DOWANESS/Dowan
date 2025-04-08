import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 数据生成函数
def make_moons_3d(n_samples=500, noise=0.1):
    """生成3D月牙形数据集"""
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  # 在第三维添加正弦变化

    # 拼接正负月牙形数据并添加噪声
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # 添加高斯噪声
    X += np.random.normal(scale=noise, size=X.shape)
    return X, y


# 2. 生成并划分数据集
# 训练数据 (1000个点)
X, y = make_moons_3d(n_samples=1000, noise=0.2)
# 测试数据 (500个点)
X_test, y_test = make_moons_3d(n_samples=500, noise=0.2)
# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# 3. 可视化原始数据
def plot_3d_data(X, y, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', marker='o')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)
    plt.show()


plot_3d_data(X_train, y_train, "训练集数据分布")
plot_3d_data(X_test, y_test, "测试集数据分布")


# 4. 模型训练与评估
def evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"\n{model_name} 验证集准确率: {acc:.4f}")
    print("分类报告:")
    print(classification_report(y_val, y_pred))
    return model, acc


# 初始化模型
models = {
    "决策树": DecisionTreeClassifier(max_depth=5, random_state=42),
    "AdaBoost+决策树": AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=3),
        n_estimators=100,
        random_state=42
    ),
    "SVM(线性核)": SVC(kernel='linear', gamma='auto', random_state=42),
    "SVM(RBF核)": SVC(kernel='rbf', gamma='auto', random_state=42),
    "SVM(多项式核)": SVC(kernel='poly', gamma='auto', degree=3, random_state=42)
}

# 训练和评估所有模型
best_model = None
best_acc = 0
results = {}

for name, model in models.items():
    trained_model, acc = evaluate_model(model, X_train, y_train, X_val, y_val, name)
    results[name] = acc
    if acc > best_acc:
        best_acc = acc
        best_model = trained_model

# 5. 在测试集上评估最佳模型
print("\n=== 在测试集上评估最佳模型 ===")
y_pred_test = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred_test)
print(f"最佳模型({list(models.keys())[list(results.values()).index(best_acc)]}) 测试集准确率: {test_acc:.4f}")
print("分类报告:")
print(classification_report(y_test, y_pred_test))


# 6. 可视化测试集分类结果
def plot_test_results(X_test, y_test, y_pred, model_name):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 正确分类的点
    correct = (y_pred == y_test)
    ax.scatter(X_test[correct, 0], X_test[correct, 1], X_test[correct, 2],
               c=y_test[correct], cmap='viridis', marker='o', label='正确分类')

    # 错误分类的点
    ax.scatter(X_test[~correct, 0], X_test[~correct, 1], X_test[~correct, 2],
               c='red', marker='x', s=100, label='错误分类')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f'测试集分类结果 ({model_name})\n准确率: {test_acc:.4f}')
    plt.legend()
    plt.show()


plot_test_results(X_test, y_test, y_pred_test,
                  f"最佳模型: {list(models.keys())[list(results.values()).index(best_acc)]}")

# 7. 模型性能比较可视化
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color=['blue', 'orange', 'green', 'red', 'purple'])
plt.ylim(0.8, 1.0)
plt.xlabel('模型')
plt.ylabel('验证集准确率')
plt.title('不同模型在验证集上的性能比较')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(results.values()):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
plt.tight_layout()
plt.show()