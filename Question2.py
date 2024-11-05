import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from scipy.optimize import differential_evolution
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import statsmodels.api as sm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 1. 数据准备
data = pd.read_excel('问题一数据集.xlsx')

# 提取 GDP 和相关指标
gdp = data['粤港澳大湾区GDP (万亿元人民币)']
population_data = data[['粤港澳大湾区人口 (百万)', '粤港澳大湾区年龄 0-14 岁 (百万)',
                         '粤港澳大湾区年龄 15-64 岁 (百万)', '粤港澳大湾区年龄 65 岁及以上 (百万)']]
technology_data = data[['粤港澳大湾区研发经费', '粤港澳大湾区专利申请数量']]
logistics_data = data[['粤港澳大湾区物流总量 (亿吨)', '粤港澳大湾区交通运输网络长度 (公里)']]
international_data = data[['全球GDP总量 (万亿美元)', '全球贸易总额 (万亿美元)',
                           '全球出口总额 (万亿美元)', '全球进口总额 (万亿美元)',
                           '全球贸易增长率 (%)']]

education_data = data[['粤港澳大湾区高中及以下教育人口',
                       '粤港澳大湾区本科及以上教育人口']]

industry_data = data[['粤港澳大湾区第一产业产值 (万亿元人民币)',
                      '粤港澳大湾区第二产业产值 (万亿元人民币)',
                      '粤港澳大湾区第三产业产值 (万亿元人民币)']]

# 选择相关因素并构建自变量
X = pd.concat([population_data, technology_data, logistics_data, international_data, education_data, industry_data], axis=1)
Y = gdp.values

# 数据标准化
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# 2. 数据分割
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
# 训练模型
# 线性回归模型
linear_model = LinearRegression().fit(X_train, Y_train)

# 随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, Y_train)


# 设置随机种子以保证结果的可重复性
random_state = 42
np.random.seed(random_state)
tf.random.set_seed(random_state)
# BP神经网络模型
nn_model = Sequential()
nn_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # 增加到64个隐层神经元
nn_model.add(Dense(32, activation='relu'))  # 添加第二个隐藏层，有32个神经元
nn_model.add(Dense(1))
nn_model.compile(optimizer='adam', loss='mean_squared_error')
nn_model.fit(X_train, Y_train, epochs=300, batch_size=16, verbose=0)  # 增加到300个epoch，batch_size为16

# 岭回归模型
ridge_model = Ridge(alpha=1.0).fit(X_train, Y_train)

# 套索回归模型
lasso_model = Lasso(alpha=0.1).fit(X_train, Y_train)

# XGBoost模型
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42).fit(X_train, Y_train)

# CatBoost模型
catboost_model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=5, random_state=42).fit(X_train, Y_train)

# AdaBoost模型
ada_model = AdaBoostRegressor(n_estimators=100, random_state=42).fit(X_train, Y_train)

# Gradient Boosting模型
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42).fit(X_train, Y_train)

# 决策树
dt_model = DecisionTreeRegressor(random_state=42).fit(X_train, Y_train)



# 计算预测值
y_pred_linear = linear_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_nn = nn_model.predict(X_test).flatten()  # 转换为一维数组
y_pred_ridge = ridge_model.predict(X_test)
y_pred_lasso = lasso_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_catboost = catboost_model.predict(X_test)
y_pred_ada = ada_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)

# 计算评价指标
def calculate_metrics(y_true, y_pred):
    R2 = r2_score(y_true, y_pred)
    MAPE = mean_absolute_percentage_error(y_true, y_pred) * 100  # 转为百分比
    RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
    return R2, MAPE, RMSE

R2_linear, MAPE_linear, RMSE_linear = calculate_metrics(Y_test, y_pred_linear)
R2_rf, MAPE_rf, RMSE_rf = calculate_metrics(Y_test, y_pred_rf)
R2_nn, MAPE_nn, RMSE_nn = calculate_metrics(Y_test, y_pred_nn)
R2_ridge, MAPE_ridge, RMSE_ridge = calculate_metrics(Y_test, y_pred_ridge)
R2_lasso, MAPE_lasso, RMSE_lasso = calculate_metrics(Y_test, y_pred_lasso)
R2_xgb, MAPE_xgb, RMSE_xgb = calculate_metrics(Y_test, y_pred_xgb)
R2_catboost, MAPE_catboost, RMSE_catboost = calculate_metrics(Y_test, y_pred_catboost)
R2_ada, MAPE_ada, RMSE_ada = calculate_metrics(Y_test, y_pred_ada)
R2_gb, MAPE_gb, RMSE_gb = calculate_metrics(Y_test, y_pred_gb)
R2_dt, MAPE_dt, RMSE_dt = calculate_metrics(Y_test, y_pred_dt)

# 将每个模型的指标存入字典
metrics = {
    'Linear Regression': {'MAPE': MAPE_linear, 'RMSE': RMSE_linear},
    'Random Forest': {'MAPE': MAPE_rf, 'RMSE': RMSE_rf},
    'BP Neural Network': {'MAPE': MAPE_nn, 'RMSE': RMSE_nn},
    'Ridge Regression': {'MAPE': MAPE_ridge, 'RMSE': RMSE_ridge},
    'Lasso Regression': {'MAPE': MAPE_lasso, 'RMSE': RMSE_lasso},
    'XGBoost': {'MAPE': MAPE_xgb, 'RMSE': RMSE_xgb},
    'CatBoost': {'MAPE': MAPE_catboost, 'RMSE': RMSE_catboost},
    'AdaBoost': {'MAPE': MAPE_ada, 'RMSE': RMSE_ada},
    'Gradient Boosting': {'MAPE': MAPE_gb, 'RMSE': RMSE_gb},
    'Decision Tree': {'MAPE': MAPE_dt, 'RMSE': RMSE_dt}
}

# 找到MAPE和RMSE的最大值，归一化
max_mape = max([metrics[model]['MAPE'] for model in metrics])
max_rmse = max([metrics[model]['RMSE'] for model in metrics])

# 计算每个模型的加权综合误差
weights = {'MAPE': 0.5, 'RMSE': 0.5}  # 可以调整权重
combined_error = {
    model: (metrics[model]['MAPE'] / max_mape) * weights['MAPE'] +
           (metrics[model]['RMSE'] / max_rmse) * weights['RMSE']
    for model in metrics
}

# 按综合误差排序
sorted_models = sorted(combined_error.items(), key=lambda x: x[1])
top_n = 5
best_models = sorted_models[:top_n]

# 输出结果
print(f"综合MAPE和RMSE最小的前 {top_n} 个模型是:")
for model, error in best_models:
    print(f"{model}: 归一化后综合误差 = {error}")


# 粒子群优化
num_weights = 3  # 权重个数
bounds = [(0, 1)] * num_weights  # 权重范围

# 定义目标函数
def objective(weights):
    weighted_pred = weights[0] * y_pred_lasso + weights[1] * y_pred_gb + weights[2] * y_pred_ridge
    return mean_squared_error(Y_test, weighted_pred)

# 运行粒子群优化
result = differential_evolution(objective, bounds, maxiter=100, popsize=50)
optimal_weights = result.x

# 计算加权预测
Y_pred_weighted = (optimal_weights[0] * y_pred_lasso +
                   optimal_weights[1] * y_pred_gb +
                   optimal_weights[2] * y_pred_ridge)

# 计算加权预测的评价指标
R2_weighted, MAPE_weighted, RMSE_weighted = calculate_metrics(Y_test, Y_pred_weighted)

print(f"加权预测的RMSE: {RMSE_weighted:.4f}")
# 可视化结果
labels1 = ['Linear Regression', 'Random Forest', 'BP Neural Network', 'Ridge Regression', 'Lasso Regression', 'XGBoost', 'CatBoost', 'AdaBoost', 'Gradient Boosting', 'Decision Tree','PSO Weighted']
r2_scores = [R2_linear, R2_rf, R2_nn, R2_ridge, R2_lasso, R2_xgb, R2_catboost, R2_ada, R2_gb, R2_dt,R2_weighted]
mape_scores = [MAPE_linear, MAPE_rf, MAPE_nn, MAPE_ridge, MAPE_lasso, MAPE_xgb, MAPE_catboost, MAPE_ada, MAPE_gb, MAPE_dt, MAPE_weighted]
rmse_scores = [RMSE_linear, RMSE_rf, RMSE_nn, RMSE_ridge, RMSE_lasso, RMSE_xgb, RMSE_catboost, RMSE_ada, RMSE_gb, RMSE_dt, RMSE_weighted]


# 定义标签列表
labels = ['真实值', '线性回归预测', '随机森林预测', 'BP神经网络预测', '岭回归预测', '套索回归预测', 
          'XGBoost预测', 'CatBoost预测', 'AdaBoost预测', '梯度提升预测', '决策树预测', 'pso加权迭代预测']

# 绘制预测模型与真实值的比对
plt.figure(figsize=(12, 6))
plt.plot(Y_test, 'o-', label=labels[0], linewidth=2)  # 真实值
plt.plot(y_pred_linear, 's-', label=labels[1], linewidth=2)  # 线性回归预测
plt.plot(y_pred_rf, 'd-', label=labels[2], linewidth=2)  # 随机森林预测
plt.plot(y_pred_nn, 'x-', label=labels[3], linewidth=2)  # BP神经网络预测
plt.plot(y_pred_ridge, 'p-', label=labels[4], linewidth=2)  # 岭回归预测
plt.plot(y_pred_lasso, 'h-', label=labels[5], linewidth=2)  # 套索回归预测
plt.plot(y_pred_xgb, 'v-', label=labels[6], linewidth=2)  # XGBoost预测
plt.plot(y_pred_catboost, '1-', label=labels[7], linewidth=2)  # CatBoost预测
plt.plot(y_pred_ada, '2-', label=labels[8], linewidth=2)  # AdaBoost预测
plt.plot(y_pred_gb, '3-', label=labels[9], linewidth=2)  # 梯度提升预测
plt.plot(y_pred_dt, '4-', label=labels[10], linewidth=2)  # 决策树预测
plt.plot(Y_pred_weighted, '^-', label=labels[11], linewidth=2)  # 加权预测

# 添加标题和标签
plt.xlabel('样本索引')
plt.ylabel('GDP (万亿元)')
plt.title('不同模型与真实值的比较')

# 显示图例
plt.legend()
plt.grid()
plt.show()



# R^2 Comparison
plt.figure(figsize=(10, 6))  # 设置宽度为10，高度为6
plt.bar(labels1, r2_scores, color='b')
plt.title('R^2 Comparison')
plt.ylabel('R^2')
plt.xticks(rotation=90)  # 使横坐标标签纵向显示
plt.tight_layout()  # 自动调整子图参数，使图表适合
plt.show()

# MAPE Comparison
plt.figure(figsize=(10, 6))
plt.bar(labels1, mape_scores, color='g')
plt.title('MAPE Comparison')
plt.ylabel('MAPE (%)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# RMSE Comparison
plt.figure(figsize=(10, 6))
plt.bar(labels1, rmse_scores, color='r')
plt.title('RMSE Comparison')
plt.ylabel('RMSE')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

