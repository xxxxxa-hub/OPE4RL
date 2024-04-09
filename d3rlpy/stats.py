from scipy.stats import norm
import numpy

# 更新的5个正态分布的均值和标准差
def get_mean_std(means, std_devs):
    # 重新计算平均分布的均值和方差
    mean_of_means = sum(means) / len(means)
    variance_of_means = sum([std_dev**2 for std_dev in std_devs]) / (len(means)**2)

    # 平均分布的标准差
    std_dev_of_means = variance_of_means**0.5

    return mean_of_means, std_dev_of_means




def get_p_value(means_1, std_devs_1, means_2, std_devs_2):
    mu1, sigma1 = get_mean_std(means_1, std_devs_1)
    mu2, sigma2 = get_mean_std(means_2, std_devs_2)
    print(mu1, sigma1)
    print(mu2, sigma2)

    # 计算差异分布的参数
    mu_Z = mu1 - mu2
    sigma_Z = (sigma1**2 + sigma2**2)**0.5

    # 计算Z-score
    Z_score = mu_Z / sigma_Z

    # 计算p值
    p_value = norm.cdf(Z_score)

    print("Z-score:", Z_score)
    print("p-value:", p_value)


def main():
    # epoch = 60
    means_1 = [1098.075, 1177.708, 1164.187, 1247.100, 1181.962]
    std_1 = [20.446, 30.500, 35.012, 28.405, 29.082]
    means_2 = [1239.301, 1230.982, 1203.047, 1304.507, 1149.060]
    std_2 = [38.455, 66.074, 51.025, 48.298, 73.211]

    # epoch = 80
    # means_1 = [1182.175, 1235.075, 1227.265, 1162.205, 1215.479]
    # std_1 = [29.246, 33.121, 29.486, 20.940, 121.350]
    # means_2 = [1380.308, 1053.945, 1321.607, 1302.515, 1289.190]
    # std_2 = [62.002, 27.331, 24.921, 67.429, 70.636]

    # epoch = 70
    # means_1 = [1172.360, 1239.737, 1324.677, 1327.332, 1218.910]
    # std_1 = [29.400, 28.706, 34.095, 26.368, 35.849]
    # means_2 = [1328.114, 1042.632, 1251.311, 1357.289, 1281.402]
    # std_2 = [35.965, 21.727, 21.900, 40.011, 33.024]

    # epoch = 90
    # means_1 = [1331.974, 1226.321, 1151.242, 1277.577, 1296.386]
    # std_1 = [46.418, 28.254, 26.058, 24.220, 29.381]
    # means_2 = [1358.911, 1161.278, 1323.793, 1248.601, 1262.859]
    # std_2 = [34.636, 27.651, 20.379, 35.181, 19.366]

    # epoch = 100
    # means_1 = [1309.502, 1242.752, 1277.104, 1350.904, 1180.292]
    # std_1 = [27.482, 29.439, 22.881, 36.680, 153.136]
    # means_2 = [1211.652, 1315.595, 1340.912, 1248.683, 1213.936]
    # std_2 = [86.680, 33.478, 47.810, 43.023, 17.301]

    get_p_value(means_1, std_1, means_2, std_2)
    print(numpy.mean(means_1))
    print(numpy.mean(means_2))
if __name__ =="__main__":
    main()