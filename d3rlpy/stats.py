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
    file_path = os.path.join(dir_path, "halfcheetah_random_0_{}.o".format(index_start))
    with open(file_path, 'r') as file:
        content = file.read()  # 读取文件所有内容
        pattern = r"Return mean when gamma = 1.0: (-?\d+\.\d+)\nReturn std when gamma = 1.0: (\d+\.\d+)"
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
    for i in range(1,100,10):
        print("Epoch:{}".format(i+109))
        process(index_start=i)
        print("-"*30)
    

if __name__ =="__main__":
    main()