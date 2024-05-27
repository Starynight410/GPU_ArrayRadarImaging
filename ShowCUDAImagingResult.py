## 绘制CUDA成像结果
# 2D成像 默认30x30
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import csv
import xlrd
import math 
import pandas as pd


## 读写csv文件
def load_csv(path):
    data_read = pd.read_csv(path, header = None)
    list = data_read.values.tolist()
    data = np.array(list)
    # 保存的每一行最后会多读1次
    new_data = data[~np. isnan (data)].reshape(data.shape[0],data.shape[1]-1) # 去除nan
    # print(new_data.shape)
    return new_data


if __name__ == '__main__':

    T = load_csv('ImagingResult.csv')
    # T = load_csv('ImagingResult3D_0524.csv')

    #  划分 方位 x 俯仰 网格
    # azi_left = -30
    # azi_right = 30
    # pit_left = -30
    # pit_right = 30
    azi_left = -60
    azi_right = 60
    pit_left = -60
    pit_right = 60

    Nay_net = round((azi_right-azi_left)*2)     # 方位向的网格数
    Naz_net = round((pit_right-pit_left)*2)     # 俯仰向的网格数

    Jd1_hori = np.zeros(Nay_net)
    Jd1_pit = np.zeros(Naz_net)

    for ii in np.arange(1,Nay_net):
        Jd1_hori[ii]=azi_left+(azi_right-azi_left)/(Nay_net-1)*(ii-1)

    for ii in np.arange(1,Nay_net):
        Jd1_pit[ii]=pit_left+(pit_right-pit_left)/(Naz_net-1)*(ii-1)

    azlabel, pitlabel = np.meshgrid(Jd1_hori,Jd1_pit)

    # Plot the surface
    fig = plt.figure()
    ax3 = plt.axes(projection = '3d')
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号
    print('成像网格:', str(T.shape))

    # ax3.plot_surface(azlabel, pitlabel, 20*np.log10(T), cmap = 'rainbow')
    ax3.plot_surface(azlabel, pitlabel, T, cmap = 'rainbow')
    ax3.view_init(elev=90., azim=0)

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0, 0)

    plt.xlim(xmin = azi_left, xmax = azi_right)
    plt.ylim(ymin = pit_left, ymax = pit_right)

    ax3.set_xlabel('Azimuth[deg]')
    ax3.set_ylabel('Elevation[deg]')
    # ax3.set_zlabel('Amplitude[dB]')
    plt.title('2D Imaging Result Using CUDA')
    plt.colorbar
    print('成像结果绘制完成\n')

    plt.show()
    
