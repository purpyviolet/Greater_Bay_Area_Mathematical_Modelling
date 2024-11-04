import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import statsmodels.api as sm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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

# 1. 读取数据
data = pd.read_excel('问题一数据集.xlsx')

# 2. 提取相关指标
population_data = data[['粤港澳大湾区人口 (百万)',
                         '粤港澳大湾区年龄 0-14 岁 (百万)',
                         '粤港澳大湾区年龄 15-64 岁 (百万)',
                         '粤港澳大湾区年龄 65 岁及以上 (百万)']].values

technology_data = data[['粤港澳大湾区研发经费',
                         '粤港澳大湾区专利申请数量']].values

logistics_data = data[['粤港澳大湾区物流总量 (亿吨)',
                       '粤港澳大湾区交通运输网络长度 (公里)']].values

international_data = data[['全球GDP总量 (万亿美元)',
                            '全球贸易总额 (万亿美元)',
                            '全球出口总额 (万亿美元)',
                            '全球进口总额 (万亿美元)',
                            '全球贸易增长率 (%)']].values

education_data = data[['粤港澳大湾区高中及以下教育人口',
                       '粤港澳大湾区本科及以上教育人口']].values

foundation_data = data[['粤港澳大湾区基础设施投资 (万亿元人民币)']].values

industry_data = data[['粤港澳大湾区第一产业产值 (万亿元人民币)',
                      '粤港澳大湾区第二产业产值 (万亿元人民币)',
                      '粤港澳大湾区第三产业产值 (万亿元人民币)']].values

gdp = data['粤港澳大湾区GDP (万亿元人民币)'].values

# 3. 国际环境指标降维（通过 PCA）
pca = PCA()
international_pca = pca.fit_transform(international_data)
international_reduced = international_pca[:, 0]  # 选择第一主成分

# # 可视化 PCA 结果
# plt.figure(figsize=(10, 6))
# plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color='b')
# plt.xlabel('主成分数量')
# plt.ylabel('累计解释方差比')
# plt.title('PCA 解释方差累积图')
# plt.grid()
# plt.show()

# 4. 人口指标使用 t-SNE 降维
n_samples = population_data.shape[0]
perplexity_value = min(30, n_samples - 1)  # Set perplexity to be less than n_samples
tsne = TSNE(perplexity=perplexity_value)
population_tsne = tsne.fit_transform(population_data)

# 5. 科技指标相关性分析
tech_corr = np.corrcoef(technology_data.T, gdp)
print(f'科技指标与 GDP 的相关系数:\n{tech_corr[-1, :-1]}')
tech_idx = np.argmax(np.abs(tech_corr[:-1, -1]))  # 找到与 GDP 相关性最大的指标
tech_reduced = technology_data[:, tech_idx]
print(f'与 GDP 相关性最大的科技指标:\n{tech_reduced}')

# scaler = MinMaxScaler()
# standardized_data = scaler.fit_transform(technology_data)

# # 计算标准化后的几何平均
# tech_reduced = np.sqrt(standardized_data[:, 0] * standardized_data[:, 1])
# print(f'与 GDP 相关的科技指标:\n{tech_reduced}')

# 6. 物流指标相关性分析
# logistics_corr = np.corrcoef(logistics_data.T, gdp)
# print(f'物流指标与 GDP 的相关系数:\n{logistics_corr[-1, :-1]}')
# logistics_idx = np.argmax(np.abs(logistics_corr[:-1, -1]))  # 找到与 GDP 相关性最大的指标
# logistics_reduced = logistics_data[:, logistics_idx]
# print(f'与 GDP 相关性最大的物流指标:\n{logistics_reduced}')

# 假设 logistics_data 包含需要的数据
logistics_volume = logistics_data[:, 0]  # 物流总量
network_length = logistics_data[:, 1]    # 运输网络长度

# 计算物流综合指数
logistics_reduced = logistics_volume * np.log(network_length)

print(f'与 GDP 相关的物流指标:\n{logistics_reduced}')

# 6.1 教育指标相关性分析
# education_corr = np.corrcoef(education_data.T, gdp)
# print(f'教育指标与 GDP 的相关系数:\n{education_corr[-1, :-1]}')
# education_idx = np.argmax(np.abs(education_corr[:-1, -1]))  # 找到与 GDP 相关性最大的指标
# education_reduced = education_data[:, education_idx]
# print(f'与 GDP 相关性最大的教育指标:\n{education_reduced}')

# 定义权重
w1 = 0.6  # 本科及以上教育人口占比的权重
w2 = 0.4  # 教育人口总量的权重

# 计算高中及以下和本科及以上教育人口的相关值
high_school_or_below = education_data[:, 0]
bachelor_or_above = education_data[:, 1]
total_population = high_school_or_below + bachelor_or_above

# 计算本科及以上教育人口占比
bachelor_ratio = bachelor_or_above / total_population

# 计算加权平均教育指数
education_reduced = (w1 * bachelor_ratio) + (w2 * total_population)

# 计算教育指数
print(f'与 GDP 相关的教育指标:\n{education_reduced}')

# 6.2 基础设施指标相关性分析
foundation_corr = np.corrcoef(foundation_data.T, gdp)
print(f'基础设施指标与 GDP 的相关系数:\n{foundation_corr[-1, :-1]}')

# 6.3 产业指标相关性分析

# 计算每个产业的权重（GDP贡献比例）假设三个产业的总价值接近gdp总量
# weights = industry_data / gdp[:, np.newaxis]
weights = industry_data / np.sum(industry_data, axis=0)

# industry_corr = np.corrcoef(industry_data.T, gdp)
# print(f'产业指标与 GDP 的相关系数:\n{industry_corr[-1, :-1]}')
# industry_idx = np.argmax(np.abs(industry_corr[:-1, -1]))  # 找到与 GDP 相关性最大的指标
# industry_reduced = industry_data[:, industry_idx]
# print(f'与 GDP 相关性最大的产业指标:\n{industry_reduced}')
# 计算CIMVI
CIMVI = np.sum(weights * industry_data, axis=1)
industry_reduced = CIMVI
print(f'与 GDP 相关的CIMVI指标:\n{industry_reduced}')

# # 7. 可视化 t-SNE 结果
# plt.figure(figsize=(10, 6))
# plt.scatter(population_tsne[:, 0], population_tsne[:, 1], marker='o')
# plt.title('人口指标的 t-SNE 降维结果')
# plt.xlabel('t-SNE 维度 1')
# plt.ylabel('t-SNE 维度 2')
# plt.grid()
# plt.show()

# 8. 准备自变量和因变量
X = np.column_stack((population_tsne, international_reduced, tech_reduced, logistics_reduced, education_reduced, industry_reduced))
# 将每一列数据标准化
scaler = MinMaxScaler()
scaler2 = StandardScaler()
standardized_population_tsne = scaler.fit_transform(population_tsne.reshape(-1, 1))
standardized_international_reduced = scaler.fit_transform(international_reduced.reshape(-1, 1))
standardized_tech_reduced = scaler.fit_transform(tech_reduced.reshape(-1, 1))
standardized_logistics_reduced = scaler2.fit_transform(logistics_reduced.reshape(-1, 1))
standardized_education_reduced = scaler.fit_transform(education_reduced.reshape(-1, 1))
standardized_industry_reduced = scaler.fit_transform(industry_reduced.reshape(-1, 1))

# 合并标准化后的数据
X = np.column_stack((
    population_tsne,
    standardized_international_reduced,
    standardized_tech_reduced,
    standardized_logistics_reduced,
    standardized_education_reduced,
    standardized_industry_reduced
))
Y = gdp

# 数据准备
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 初始化模型
models = {
    '线性回归': LinearRegression(),
    '岭回归': Ridge(),
    '套索回归': Lasso(),
    '支持向量机': SVR(),
    '决策树': DecisionTreeRegressor(),
    '随机森林': RandomForestRegressor(),
    '梯度提升树': GradientBoostingRegressor(),
    '神经网络': MLPRegressor(max_iter=1000),
    'XGBoost': XGBRegressor(),
    'CatBoost': CatBoostRegressor(silent=True),
    'AdaBoost': AdaBoostRegressor(),
}

# 存储预测结果和评价指标
results = {}
metrics = {}

# 逐个训练和预测
for name, model in models.items():
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    # print(len(predictions), len(Y_test))

    # 存储预测结果
    results[name] = predictions

    # 计算评价指标
    mse = mean_squared_error(Y_test, predictions)
    r2 = r2_score(Y_test, predictions)
    metrics[name] = {'MSE': mse, 'R2': r2}


print(len(X_test), len(Y_test))
# 假设有5个样本数，因此定义5个年份
years = [2020, 2021, 2022, 2023, 2024]

plt.figure(figsize=(12, 6))
plt.plot(years, Y_test, 'r--', label='真实值')  # 用年份作为 x 轴

# 遍历模型的预测结果并绘制
for name, preds in results.items():
    plt.plot(years, preds, label=name)  # 使用 years 作为 x 轴标签

# 设置标题、标签和图例
plt.title('各模型预测结果对比')
plt.xlabel('年份')
plt.ylabel('预测值')
plt.legend()
plt.grid()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# 假设 metrics 是一个字典，你已经创建了 metrics_df
metrics_df = pd.DataFrame(metrics).T
ax = metrics_df.plot(kind='bar', figsize=(12, 6))
plt.title('模型评价指标对比')
plt.ylabel('指标值')
plt.xticks(rotation=45)
plt.grid()

# 添加柱形图上方的值
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=6)

plt.show()


for name, preds in results.items():
    print(name,metrics[name])
