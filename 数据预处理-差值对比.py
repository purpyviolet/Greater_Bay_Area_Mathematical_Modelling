# 数据输入
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 年份
year = np.arange(2000, 2025)

# 粤港澳大湾区GDP (万亿元人民币)，缺失值用 NaN 表示
gdp = np.array([1.24, 1.37, 1.52, 1.72, 2.00, 2.29, 2.70, 3.38, 3.99, 4.37, np.nan, 6.12, 6.73, 7.44, 8.05, 8.61, 9.33, 10.26, np.nan, 12.38, 12.67, 13.34, 14.3, 15.12, 16.05])
# 东京第一产业产值 (万亿日元)，缺失值用 NaN 表示
tokyo_agri = np.array([3.1375, 3.1875, 3.2125, np.nan, 3.2875, 3.3375, 3.3750, 3.4060, 3.4375, 3.4560, 3.4690, 3.4875, 3.5000, 3.5190, np.nan, 3.5440, 3.5625, 3.5750, 3.5875, 3.6000, 3.6250, 3.6375, 3.6500, 3.6625, 3.6750])

# 线性插值
gdp_interp = np.copy(gdp)
nan_indices_gdp = np.isnan(gdp)
gdp_interp[nan_indices_gdp] = np.interp(year[nan_indices_gdp], year[~nan_indices_gdp], gdp[~nan_indices_gdp])

tokyo_agri_interp = np.copy(tokyo_agri)
nan_indices_tokyo_agri = np.isnan(tokyo_agri)
tokyo_agri_interp[nan_indices_tokyo_agri] = np.interp(year[nan_indices_tokyo_agri], year[~nan_indices_tokyo_agri], tokyo_agri[~nan_indices_tokyo_agri])

# 绘图
plt.figure(figsize=(10, 8))

# 粤港澳大湾区GDP对比
plt.subplot(2, 1, 1)
plt.plot(year, gdp, 'ro-', linewidth=1.5, label='原始数据')
plt.plot(year, gdp_interp, 'b*-', linewidth=1.5, label='插值数据')
plt.grid(True)
plt.xlabel('年份')
plt.ylabel('GDP (万亿元人民币)')
plt.legend()

# 东京第一产业产值对比
plt.subplot(2, 1, 2)
plt.plot(year, tokyo_agri, 'ro-', linewidth=1.5, label='原始数据')
plt.plot(year, tokyo_agri_interp, 'b*-', linewidth=1.5, label='插值数据')
plt.grid(True)
plt.title('东京第一产业产值 插值前后对比')
plt.xlabel('年份')
plt.ylabel('第一产业产值 (万亿日元)')
plt.legend()

plt.tight_layout()
plt.show()