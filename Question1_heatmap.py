import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 1. 数据准备
data = pd.read_excel('粤港澳数据集.xlsx')
# 1. 创建列名缩写映射
rename_columns = {
    '粤港澳大湾区GDP (万亿元人民币)': 'GDP',
    '粤港澳大湾区人口 (百万)': '人口',
    '粤港澳大湾区年龄 0-14 岁 (百万)': '年龄 0-14',
    '粤港澳大湾区年龄 15-64 岁 (百万)': '年龄 15-64',
    '粤港澳大湾区年龄 65 岁及以上 (百万)': '年龄 65+',
    '粤港澳大湾区研发经费': '研发经费',
    '粤港澳大湾区专利申请数量': '专利申请',
    '粤港澳大湾区物流总量 (亿吨)': '物流总量',
    '粤港澳大湾区交通运输网络长度 (公里)': '交通网络长度',
    '全球GDP总量 (万亿美元)': '全球GDP',
    '全球贸易总额 (万亿美元)': '全球贸易总额',
    '全球出口总额 (万亿美元)': '全球出口',
    '全球进口总额 (万亿美元)': '全球进口',
    '全球贸易增长率 (%)': '全球贸易增长率',
    '粤港澳大湾区高中及以下教育人口': '高中及以下教育',
    '粤港澳大湾区本科及以上教育人口': '本科及以上教育',
    '粤港澳大湾区基础设施投资 (万亿元人民币)': '基础设施投资',
    '粤港澳大湾区第一产业产值 (万亿元人民币)': '第一产业',
    '粤港澳大湾区第二产业产值 (万亿元人民币)': '第二产业',
    '粤港澳大湾区第三产业产值 (万亿元人民币)': '第三产业'
}

# 2. 创建一个带有简化列名的数据副本
data_renamed = data.rename(columns=rename_columns)

# 3. 提取重命名后的相关数据
gdp = data_renamed['GDP']
population_data = data_renamed[['人口', '年龄 0-14', '年龄 15-64', '年龄 65+']]
technology_data = data_renamed[['研发经费', '专利申请']]
logistics_data = data_renamed[['物流总量', '交通网络长度']]
international_data = data_renamed[['全球GDP', '全球贸易总额', '全球出口', '全球进口', '全球贸易增长率']]
education_data = data_renamed[['高中及以下教育', '本科及以上教育']]
industry_data = data_renamed[['第一产业', '第二产业', '第三产业']]
# # 提取 GDP 和相关指标
# gdp = data['粤港澳大湾区GDP (万亿元人民币)']
# population_data = data[['粤港澳大湾区人口 (百万)',
#                          '粤港澳大湾区年龄 0-14 岁 (百万)',
#                          '粤港澳大湾区年龄 15-64 岁 (百万)',
#                          '粤港澳大湾区年龄 65 岁及以上 (百万)']]

# technology_data = data[['粤港澳大湾区研发经费', '粤港澳大湾区专利申请数量']]
# logistics_data = data[['粤港澳大湾区物流总量 (亿吨)',
#                         '粤港澳大湾区交通运输网络长度 (公里)']]
# international_data = data[['全球GDP总量 (万亿美元)',
#                             '全球贸易总额 (万亿美元)',
#                             '全球出口总额 (万亿美元)',
#                             '全球进口总额 (万亿美元)',
#                             '全球贸易增长率 (%)']]

# education_data = data[['粤港澳大湾区高中及以下教育人口',
#                        '粤港澳大湾区本科及以上教育人口']]

# foundation_data = data[['粤港澳大湾区基础设施投资 (万亿元人民币)']]

# industry_data = data[['粤港澳大湾区第一产业产值 (万亿元人民币)',
#                       '粤港澳大湾区第二产业产值 (万亿元人民币)',
#                       '粤港澳大湾区第三产业产值 (万亿元人民币)']]

# 2. 数据探索与预处理
print(data.describe())

# 3. 相关性分析
corr_matrix = pd.concat([gdp, population_data, technology_data, logistics_data, international_data, education_data, industry_data], axis=1).corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('相关性热图',fontsize=20)
#plt.xlabel('因素',fontsize=18)
#plt.ylabel('GDP',fontsize=18)
plt.xticks(rotation=90, fontsize=12)  # 修改横轴标签字体大小和旋转角度
plt.yticks(rotation=0, fontsize=12)   # 修改纵轴标签字体大小
# 保存图像
plt.savefig('Q1/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()



# 4. 主成分分析（针对国际环境）
pca = PCA()
pca.fit(international_data)
explained_variance = pca.explained_variance_ratio_ * 100  # 解释的方差百分比

# 可视化主成分解释的方差
plt.figure()
plt.bar(range(1, len(explained_variance) + 1), explained_variance)
plt.xlabel('主成分')
plt.ylabel('解释的方差百分比')
plt.title('主成分分析的方差解释')
plt.show()

# 选择第一主成分作为国际环境的代表
international_reduced = pca.transform(international_data)[:, 0]

# 5. 科技和物流指标相关性分析，选择相关性最大的
tech_corr = technology_data.corrwith(gdp)
tech_idx = tech_corr.abs().idxmax()  # 找到与 GDP 相关性最大的指标
tech_reduced = technology_data[tech_idx] # 使用列名直接索引

logistics_corr = logistics_data.corrwith(gdp)
logistics_idx = logistics_corr.abs().idxmax()  # 找到与 GDP 相关性最大的指标
logistics_reduced = logistics_data[logistics_idx]  # 使用列名直接索引

# 6.1 教育指标相关性分析
education_corr = education_data.corrwith(gdp)
education_idx = education_corr.abs().idxmax()  # 找到与 GDP 相关性最大的指标
education_reduced = education_data[education_idx]  # 使用列名直接索引

# 6.2 产业指标相关性分析
industry_corr = industry_data.corrwith(gdp)
industry_idx = industry_corr.abs().idxmax()  # 找到与 GDP 相关性最大的指标
industry_reduced = industry_data[industry_idx]  # 使用列名直接索引


# 6. 准备自变量和因变量
X = pd.concat([population_data,
                pd.Series(international_reduced, name='国际环境主成分'),
                tech_reduced,
                logistics_reduced,
                education_reduced,
                industry_reduced], axis=1)

# 初始化 MinMaxScaler
scaler = MinMaxScaler()

# 对每个数据集进行标准化
standardized_population_data = pd.DataFrame(scaler.fit_transform(population_data), columns=population_data.columns)
standardized_international_reduced = pd.Series(scaler.fit_transform(international_reduced.reshape(-1, 1)).flatten(), name='国际环境主成分')
standardized_tech_reduced = pd.Series(scaler.fit_transform(tech_reduced.values.reshape(-1, 1)).flatten(), name='技术指数')
standardized_logistics_reduced = pd.Series(scaler.fit_transform(logistics_reduced.values.reshape(-1, 1)).flatten(), name='物流指数')
standardized_education_reduced = pd.Series(scaler.fit_transform(education_reduced.values.reshape(-1, 1)).flatten(), name='教育指数')
standardized_industry_reduced = pd.Series(scaler.fit_transform(industry_reduced.values.reshape(-1, 1)).flatten(), name='产业指数')



# 6. 准备自变量和因变量
X = pd.concat([standardized_population_data,
                pd.Series(international_reduced, name='国际环境主成分'),
                standardized_tech_reduced,
                standardized_logistics_reduced,
                standardized_education_reduced,
                standardized_industry_reduced], axis=1)

Y = gdp

# 添加常数项
X = np.c_[np.ones(X.shape[0]), X]

# 7. 建立多元线性回归模型
b = np.linalg.lstsq(X, Y, rcond=None)[0]

# 显示回归系数
print('线性回归系数:')
print(b)

# 计算 R 平方值
Y_hat = X.dot(b)  # 预测值
SS_tot = np.sum((Y - np.mean(Y))**2)  # 总平方和
SS_res = np.sum((Y - Y_hat)**2)  # 残差平方和
R_squared = 1 - (SS_res / SS_tot)
print(f'R平方值: {R_squared:.4f}')

# 8. 重要因素选择与分级
print('重要因素分析:')
for i in range(1, len(b)):
    if abs(b[i]) > 0.5:  # 根据阈值判断关键因素
        print(f'因素 {i} 是关键因素: {b[i]:.4f}')
    else:
        print(f'因素 {i} 是次要因素: {b[i]:.4f}')

# 9. 可视化回归系数
plt.figure()
plt.bar(range(len(b[1:])), b[1:], color='skyblue')  # 不包括常数项
plt.xlabel('因素')
plt.ylabel('回归系数')
plt.title('各因素对 GDP 的影响')
plt.xticks(range(len(b[1:])), ['人口', '年龄 0-14', '年龄 15-64', '年龄 65+', '国际环境主成分', '科技指标', '物流指标', '教育指标', '产业指标'])
plt.show()
