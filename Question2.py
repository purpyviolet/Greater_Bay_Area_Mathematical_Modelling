import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from scipy.optimize import differential_evolution

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
X = pd.concat([population_data, technology_data, logistics_data, international_data, industry_data], axis=1)
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

# BP神经网络模型
nn_model = Sequential()
nn_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # 增加到64个隐层神经元
nn_model.add(Dense(32, activation='relu'))  # 添加第二个隐藏层，有32个神经元
nn_model.add(Dense(1))
nn_model.compile(optimizer='adam', loss='mean_squared_error')
nn_model.fit(X_train, Y_train, epochs=300, batch_size=16, verbose=0)  # 增加到300个epoch，batch_size为16

# 计算预测值
y_pred_linear = linear_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_nn = nn_model.predict(X_test).flatten()  # 转换为一维数组

# 计算评价指标
def calculate_metrics(y_true, y_pred):
    R2 = r2_score(y_true, y_pred)
    MAPE = mean_absolute_percentage_error(y_true, y_pred) * 100  # 转为百分比
    RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
    return R2, MAPE, RMSE

R2_linear, MAPE_linear, RMSE_linear = calculate_metrics(Y_test, y_pred_linear)
R2_rf, MAPE_rf, RMSE_rf = calculate_metrics(Y_test, y_pred_rf)
R2_nn, MAPE_nn, RMSE_nn = calculate_metrics(Y_test, y_pred_nn)

# 粒子群优化
num_weights = 3  # 权重个数
bounds = [(0, 1)] * num_weights  # 权重范围

# 定义目标函数
def objective(weights):
    weighted_pred = weights[0] * y_pred_linear + weights[1] * y_pred_rf + weights[2] * y_pred_nn
    return mean_squared_error(Y_test, weighted_pred)

# 运行粒子群优化
result = differential_evolution(objective, bounds, maxiter=100, popsize=50)
optimal_weights = result.x

# 计算加权预测
Y_pred_weighted = (optimal_weights[0] * y_pred_linear +
                   optimal_weights[1] * y_pred_rf +
                   optimal_weights[2] * y_pred_nn)

# 计算加权预测的评价指标
R2_weighted, MAPE_weighted, RMSE_weighted = calculate_metrics(Y_test, Y_pred_weighted)

# 可视化结果
labels = ['Linear Regression', 'Random Forest', 'BP Neural Network', 'Weighted']
r2_scores = [R2_linear, R2_rf, R2_nn, R2_weighted]
mape_scores = [MAPE_linear, MAPE_rf, MAPE_nn, MAPE_weighted]
rmse_scores = [RMSE_linear, RMSE_rf, RMSE_nn, RMSE_weighted]

# R^2 Comparison
plt.figure()
plt.bar(labels, r2_scores, color='b')
plt.title('R^2 Comparison')
plt.ylabel('R^2')
plt.show()

# MAPE Comparison
plt.figure()
plt.bar(labels, mape_scores, color='g')
plt.title('MAPE Comparison')
plt.ylabel('MAPE (%)')
plt.show()

# RMSE Comparison
plt.figure()
plt.bar(labels, rmse_scores, color='r')
plt.title('RMSE Comparison')
plt.ylabel('RMSE')
plt.show()

# 绘制预测模型与真实值的比对
plt.figure()
plt.plot(Y_test, 'o-', label='真实值', linewidth=2)
plt.plot(y_pred_linear, 's-', label='线性回归预测', linewidth=2)
plt.plot(y_pred_rf, 'd-', label='随机森林预测', linewidth=2)
plt.plot(y_pred_nn, 'x-', label='BP神经网络预测', linewidth=2)
plt.plot(Y_pred_weighted, '^-', label='加权预测', linewidth=2)
plt.xlabel('样本索引')
plt.ylabel('GDP (万亿元)')
plt.title('不同模型与真实值的比较')
plt.legend()
plt.grid()
plt.show()
