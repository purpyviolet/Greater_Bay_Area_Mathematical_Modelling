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