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

# 可视化 PCA 结果
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color='b')
plt.xlabel('主成分数量')
plt.ylabel('累计解释方差比')
plt.title('PCA 解释方差累积图')
plt.grid()
plt.show()

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

# 7. 可视化 t-SNE 结果
plt.figure(figsize=(10, 6))
plt.scatter(population_tsne[:, 0], population_tsne[:, 1], marker='o')
plt.title('人口指标的 t-SNE 降维结果')
plt.xlabel('t-SNE 维度 1')
plt.ylabel('t-SNE 维度 2')
plt.grid()
plt.show()

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

# 9. 添加常数项
X = sm.add_constant(X)

# 10. 进行多元线性回归
model = sm.OLS(Y, X).fit()

# 11. 显示结果
print('线性回归系数:')
print(model.params)

# 12. 计算R平方值
R_squared = model.rsquared
print(f'R平方值: {R_squared:.4f}')

# 13. 可视化回归结果
Y_hat = model.predict(X)
plt.figure(figsize=(10, 6))
plt.scatter(Y, Y_hat, edgecolor='k', alpha=0.7)
plt.xlabel('实际 GDP')
plt.ylabel('预测 GDP')
plt.title('实际 GDP vs 预测 GDP')
plt.plot([min(Y), max(Y)], [min(Y), max(Y)], 'r--', linewidth=2)  # 45度线
plt.legend(['预测结果', '45度线']) # 这里有错误要修改
plt.grid()
plt.show()

# 14. 残差分析
residuals = Y - Y_hat

# 残差图
plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
plt.scatter(Y_hat, residuals, edgecolor='k', alpha=0.7)
plt.xlabel('预测 GDP')
plt.ylabel('残差')
plt.title('预测 GDP 与残差')
plt.axhline(0, color='r', linestyle='--', linewidth=2)
plt.grid()

# QQ图
plt.subplot(2, 1, 2)
sm.qqplot(residuals, line='s', ax=plt.gca())
plt.title('残差的 QQ 图')
plt.tight_layout()
plt.show()

# 15. 相关性热图
# 计算自变量与因变量的相关性
if X.shape[0] == Y.shape[0]:  # Ensure the number of samples matches
    corr_matrix = np.corrcoef(X[:, 1:], Y, rowvar=False)  # rowvar=False indicates that rows are variables
else:
    raise ValueError("X and Y must have the same number of samples.")

# 可视化相关性热图
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix[:-1, -1].reshape(1, -1), annot=True, cmap='coolwarm', cbar=True,
            xticklabels=['t-SNE', '国际环境主成分', '科技指标', '物流指标', '教育指标', '产业指标'], yticklabels=['GDP'])
plt.title('自变量与 GDP 的相关性热图')
plt.show()

# 16. 回归系数的可视化
plt.figure(figsize=(10, 6))
plt.bar(range(len(model.params[1:])), model.params[1:], color='c', alpha=0.7)
plt.xticks(range(len(model.params[1:])), ['t-SNE 维度 1', 't-SNE 维度 2', '国际环境主成分', '科技指标', '物流指标', '教育指标', '产业指标'], rotation=45)
plt.xlabel('变量')
plt.ylabel('回归系数')
plt.title('回归系数可视化')
plt.grid()
plt.show()
