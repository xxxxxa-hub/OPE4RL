from scipy.stats import norm
import numpy
from statistics import mean, stdev
import os
import pdb
import re


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


def get_mean_std_list(dir_path, index_start):
    file_path = os.path.join(dir_path, "ops_0_{}.o".format(index_start))
    with open(file_path, 'r') as file:
        content = file.read()  # 读取文件所有内容
        pattern = r"Return mean when gamma = 1.0: (\d+\.\d+)\nReturn std when gamma = 1.0: (\d+\.\d+)"
        mean, std = re.findall(pattern, content)[0]
    return round(float(mean), 3), round(float(std), 3)


def process(dir_path = "/home/xiaoan/OPE4RL/d3rlpy/.onager/logs/gaoqitong-exxact", 
            index_start = 1):

    means_1 = []
    std_1 = []
    means_2 = []
    std_2 = []

    for i in range(index_start, index_start + 10 ,2):
        mean, std = get_mean_std_list(dir_path, i)
        means_1.append(mean)
        std_1.append(std)
    
    for i in range(index_start + 1, index_start + 11, 2):
        mean, std = get_mean_std_list(dir_path, i)
        means_2.append(mean)
        std_2.append(std)

    get_p_value(means_1, std_1, means_2, std_2)

def main():
    # halfcheetah-medium-replay-v0 100 200 1 abs no-clip decay
    # Epoch 110
    # means_1 = [1026.294, 985.780, 1037.592, 999.773, 1030.151]
    # std_1 = [41.198, 25.288, 21.621, 18.914, 33.439]
    # means_2 = [1038.318, 1050.075, 1053.654, 1013.114, 1046.343]
    # std_2 = [23.797, 28.678, 23.739, 20.936, 22.270]

    # Epoch 120
    # means_1 = [1068.838, 1029.405, 994.092, 1006.035, 1065.968]
    # std_1 = [22.248, 22.357, 27.002, 24.742, 27.699]
    # means_2 = [1075.933, 1047.008, 1047.118, 1036.374, 1037.433]
    # std_2 = [22.924, 24.018, 20.496, 31.950, 24.185]

    # Epoch 130
    # means_1 = [1035.069, 1018.000, 1023.716, 1004.055, 1044.876]
    # std_1 = [19.922, 21.344, 20.756, 21.297, 22.947]
    # means_2 = [1074.580, 1052.417, 1071.788, 1051.097, 1054.403]
    # std_2 = [25.769, 41.927, 29.753, 25.212, 21.743]

    # Epoch 140
    # means_1 = [1077.480, 1043.900, 1050.695, 1015.722, 1039.474]
    # std_1 = [41.961, 23.964, 36.448, 24.945, 23.748]
    # means_2 = [1079.939, 1054.581, 1057.848, 1057.510, 1063.571]
    # std_2 = [23.949,23.681, 35.020, 23.117, 21.339]

    # get_p_value(means_1, std_1, means_2, std_2)

    process(index_start=51)
    

if __name__ =="__main__":
    main()