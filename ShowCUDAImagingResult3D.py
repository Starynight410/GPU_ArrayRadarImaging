## 绘制CUDA成像结果
# 3D成像 默认30x30, CUDA程序: ArrayImaging3D.cu
# 按列将空间网格重排,绘制三维结果
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import csv
import xlrd
import math 
import pandas as pd
 
c = 3E8
B_real = 1.75E9

## 读写csv文件
def load_csv(path):
    data_read = pd.read_csv(path, header = None)
    list = data_read.values.tolist()
    data = np.array(list)
    return data


if __name__ == '__main__':

    # T = load_csv('ImagingResult.csv')
    T = load_csv('CFARImagingResult3D_0524.csv')    # 包含多个距离切片

    #  划分 方位 x 俯仰 网格
    azi_left = -30
    azi_right = 30
    pit_left = -30
    pit_right = 30
    # azi_left = -60
    # azi_right = 60
    # pit_left = -60
    # pit_right = 60

    Nay_net = round((azi_right-azi_left)*2)     # 方位向的网格数
    Naz_net = round((pit_right-pit_left)*2)     # 俯仰向的网格数
    Nax_net = (int)(np.size(T, 0) / Nay_net)    # 距离向的网格数, 每Nay_net*Nax_net个点为一组划分

    # 划分距离维网格
    Rtar = 0.5     # m
    Nax_net_div = c/2/B_real

    Rtar_s = np.zeros(Nax_net)
    for ii in np.arange(0,Nax_net):
        Rtar_s[ii] = -(int)(Nax_net/2)*Nax_net_div + Rtar + ii*Nax_net_div;  
        print(Rtar_s[ii])

    Jd1_hori = np.zeros(Nay_net)
    Jd1_pit = np.zeros(Naz_net*Nax_net)

    for ii in np.arange(1,Nay_net):
        Jd1_hori[ii]=azi_left+(azi_right-azi_left)/(Nay_net-1)*(ii-1)

    for ii in np.arange(1,Naz_net*Nax_net):
        Jd1_pit[ii]=pit_left+(pit_right-pit_left)/(Naz_net-1)*(ii-1)

    azlabel, pitlabel = np.meshgrid(Jd1_hori,Jd1_pit)

    # 获取目标点三维坐标
    idx = np.array(np.where(T==1))  # 获取索引
    # if idx.shape[1]>0:
    #     print('where T==1, index is: ')
    #     for i in range(idx.shape[1]):
    #         print((idx[0,i],idx[1,i]))
    # else:
    #     print('no exits index where T==1')

    Target_r = np.zeros(idx.shape[1])
    Target_azi = np.zeros(idx.shape[1])
    Target_ele = np.zeros(idx.shape[1])

    for ii in range(idx.shape[1]):    # 这里直接认为azi和ele维度相同
        Target_azi[ii] = Jd1_hori[idx[0,ii]%Nay_net]
        Target_ele[ii] = Jd1_pit[idx[1,ii]]
        Target_r[ii] = Rtar_s[(int)(idx[0,ii]/Nay_net)]

    # Plot the point cloud
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(Target_r, Target_azi, Target_ele)

    # 设置三维图图形区域背景颜色（r,g,b,a）
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0, 0)

    ax.set_xlim(xmin = Rtar_s[0], xmax = Rtar_s[Nax_net-1])
    ax.set_ylim(ymin = azi_left, ymax = azi_right)
    ax.set_zlim(zmin = pit_left, zmax = pit_right)

    ax.set_xlabel('Range[m]')
    ax.set_ylabel('Azimuth[deg]')
    ax.set_zlabel('Elevation[deg]')

    plt.title('3D Point Cloud Imaging Result Using CUDA')
    plt.colorbar
    print('成像结果绘制完成\n')

    plt.show()
    
