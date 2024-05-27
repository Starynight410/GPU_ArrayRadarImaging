# GPU_ArrayRadarImaging
利用CUDA加速阵列雷达3D成像过程，与传统方法对比。

# 开发环境
Linux Ubuntu 18.04, CUDA 12.0

编译运行：

nvcc -o ArrayImaging ArrayImaging.cu --ptxas-options=-v --use_fast_math -lcublas -lcufft

./ArrayImaging

# 成像结果
流程：误差校准 + range_fft + 2DBP

雷达采集1帧数据为：8路并行，512 点, 32 Chirp, 32个收发通道

测试目标角反位于50cm，方位俯仰角大约0°处，成像角度范围为-30°~30°。点目标成像结果:

Version 1

耗时：11.1002s (-30,30)

![image]{image/50cm00.jpg}
