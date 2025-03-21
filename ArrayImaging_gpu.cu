/*****************************************************************************

    Project: 综合孔径阵列雷达成像_gpu 2024_12_11
    ================

    读据本地雷达采集数据,收发阵元位置,校准权重,输出2D成像结果

    load local 8xOSPI data

    实现方式
    ================

    通过GPU并行加速计算成像结果

    gpu:成像和计算A_Comp补偿因子均划分Na个Grid,每个Grid包含Naz_net*Nay_net的二维block

    Compiling the program
    ===================

    Type `make` to compile the program. Alternatively, type the following commands:

    nvcc -o ArrayImaging_gpu ArrayImaging_gpu.cu --ptxas-options=-v --use_fast_math -lcublas -lcufft
    ./ArrayImaging_gpu
   
****************************************************************************/
#include "cufft.h"
#include "readCSV.h"
#include "readRadarData.h"
#include <math.h>
#include <complex.h>
#include <time.h> 
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdio.h>

#pragma comment(lib,"winmm.lib")

//#include"fftw3.h"
#pragma comment(lib, "libfftw3-3.lib") // double版本
// #pragma comment(lib, "libfftw3f-3.lib")// float版本
// #pragma comment(lib, "libfftw3l-3.lib")// long double版本

#define CHECK(call)\
{\
    if ((call) != cudaSuccess)\
            {\
        printf("Error: %s:%d, ", __FILE__, __LINE__);\
        printf("code:%d, reason: %s\n", (call), cudaGetErrorString(cudaGetLastError()));\
        exit(1);\
            }\
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600 

#else 
   __device__ double atomicAdd(double* a, double b) 
   { 
      return b; 
   } 
#endif

const double PI = 3.141592653;

// param
const double    c = 3E8;
const double    fc = 7.5E10;
const double    lambda = c/fc;
const double    k0 = 2*PI/lambda;
const int       Head = 0;
const int       numByte = 2;
const int       Num_Tx = 8;
const int       Num_RX = 4;
const int       NumANT = Num_RX*Num_Tx*8;
// const int       numANT = 256;   // 临时
const int       AziANT = 64;
const int       PitANT = 12;
const int       numChirp = 32;
const int       num_sample_real = 512;
const double    fs_real = 20E6;
const double    B_real = 1.75E9;
const double    ramp_end_time = 30E-6;
double          numPT = (double)num_sample_real;
double          Kr = B_real/ramp_end_time;
double          Br = num_sample_real/fs_real*Kr;
double          R_res = c/2/Br;
double          R_max = R_res*numPT;
double          PRT = 10E-6+ramp_end_time+5E-6;
double          len = numPT*numChirp*NumANT*numByte;

const int       tx_PerArray = 4;
const int       rx_PerArray = 4;
int             numArray = tx_PerArray*16;  // 64,这里没对收发不同的情况做声明

// 解包参数
struct commonCfg{
    int numByte = 2;
    int numSamp = 512;
    int numChirp = 32;
    int numTx = 4;
    int numRx = 8;
    int numANT = numTx*numRx*8; // 解包合并读8路bin
}commonCfgStruct;

// 函数声明
void DataPacUart(int *RecDataBuff, int ***ADCBuf, struct commonCfg commonCfgStruct);
int maxSearch(double a[], int n);

// 核函数
__global__ void computeAComp(
    cufftDoubleComplex *A_Comp, double *txPos_full_x, double *txPos_full_y, double *rxPos_full_x, double *rxPos_full_y, \
    int *txnum, int *rxnum, int *blank_num, double lambda, double *Jd_hori, double *Jd_pit, \
    double Rtar, double k0, int Na, int Naz_net, int Nay_net, int tx_PerArray);

__global__ void calculateBrightness(
    cuDoubleComplex *A_Comp, cuDoubleComplex *R_sigdata, double *T_sence_mo, \
    int Na, int Naz_net, int Nay_net);

// ===========================main===========================
int main()
{
    int begintime, endtime;

    // read binFile
    // char filePath[] = "./第一组_泡沫垫高A/Target50cm0d0d_8MCU.bin";  // 8路采集合并, size=1048576*8
    // char filePath[] = "/home/nx/CUDA_RadarImaging/nvvp_workspace/Cuda3DImaging_Project/第二组_双目标泡沫垫高A/TwoTargets50cm0d10d_8MCU.bin";  // 8路采集合并, size=1048576*8
    char filePath[] = "/home/nx/CUDA_RadarImaging/nvvp_workspace/Cuda3DImaging_Project/第一组_泡沫垫高A/Target50cm0d0d_8MCU.bin";  // 8路采集合并, size=1048576*8

    int size = getBinSize(filePath);

    unsigned char *buf = NULL;
    buf = (unsigned char*)malloc(size * sizeof(unsigned char));
    readBin(filePath, buf, size);
    unsigned char *fbuf = (unsigned char*)buf;

    int *RecData = NULL;
	RecData = (int *)malloc(size * sizeof(int)); // 接收数据 

	for(int i = 0; i < size; i++)
    {
        RecData[i] = *(fbuf + i);
    }  

    if (buf != NULL)
	{
		free(buf);
		buf = NULL;
	}

    // free(fbuf);
    
    int frameNum = floor(size/(len+Head));  // 8

    // 动态分配三维数组的内存  
    int ***ADCBuf = (int ***)malloc(num_sample_real * sizeof(int **));  
    if (ADCBuf == NULL) {  
        perror("Failed to allocate memory for ADCBuf array");  
        return EXIT_FAILURE;  
    }  
  
    for (int i = 0; i < num_sample_real; i++) {  
        ADCBuf[i] = (int **)malloc(numChirp * sizeof(int *));  
        if (ADCBuf[i] == NULL) {  
            perror("Failed to allocate memory for a layer of people");  
            // 释放已经分配的内存  
            for (int j = 0; j < i; j++) {  
                free(ADCBuf[j]);  
            }  
            free(ADCBuf);  
            return EXIT_FAILURE;  
        }  
  
        for (int j = 0; j < numChirp; j++) {  
            ADCBuf[i][j] = (int *)malloc(NumANT * 8 * sizeof(int));  
            if (ADCBuf[i][j] == NULL) {  
                perror("Failed to allocate memory for a row of people");  
                // 释放已经分配的内存  
                for (int k = 0; k < j; k++) {  
                    free(ADCBuf[i][k]);  
                }  
                free(ADCBuf[i]);  
                // 释放之前已经分配的内存  
                for (int l = 0; l < i; l++) {  
                    free(ADCBuf[l]);  
                }  
                free(ADCBuf);  
                return EXIT_FAILURE;  
            }  
        }  
    }  
    // 解包
    DataPacUart(RecData, ADCBuf, commonCfgStruct);
    if (RecData != NULL)
	{
		free(RecData);
		RecData = NULL;
	}
    // printf("%d\n",ADCBuf[0][0][0]);
    
    // ===========================Proccess Calibrated Data===========================
    // 读取本地校准权重 1 x Na complex double
    char filename_overallCal[] = "/home/nx/CUDA_RadarImaging/nvvp_workspace/Cuda3DImaging_Project/Target50m0d0d_overallCal.csv";
    char line[4096];
    double **overallCal;
    overallCal = readCSV(filename_overallCal, line, overallCal); // 实部:overallCal[0][i] 虚部:overallCal[1][i]
    
    // 获取阵元位置坐标
    char filename_txPos[] = "/home/nx/CUDA_RadarImaging/nvvp_workspace/Cuda3DImaging_Project/txPos_full.csv";
    char filename_rxPos[] = "/home/nx/CUDA_RadarImaging/nvvp_workspace/Cuda3DImaging_Project/rxPos_full.csv";

    // char line[1024];
    double **txPos_full;
    double **rxPos_full;
    txPos_full = readCSV(filename_txPos, line, txPos_full);
    rxPos_full = readCSV(filename_rxPos, line, rxPos_full);
    double txPos_full_x[numArray];
    double txPos_full_y[numArray]; 
    double rxPos_full_x[numArray];
    double rxPos_full_y[numArray]; 

    // 提取txPos_full的第一行到txPos_x
    for (int i = 0; i < numArray; i++) {
        txPos_full_x[i] = txPos_full[0][i];
    }
    // 提取txPos_full的第二行到txPos_y
    for (int i = 0; i < numArray; i++) {
        txPos_full_y[i] = txPos_full[1][i];
    }
    // 提取rxPos_full的第一行到txPos_x
    for (int i = 0; i < numArray; i++) {
        rxPos_full_x[i] = rxPos_full[0][i];
    }
    // 提取rxPos_full的第二行到txPos_y
    for (int i = 0; i < numArray; i++) {
        rxPos_full_y[i] = rxPos_full[1][i];
    }

    // ===========================1d range fft===========================
    const int nfft_r = 512;
    const int BATCH = 1;
    double aaa = 1.0/nfft_r; //fft系数

    cufftHandle plan;  // 创建句柄
    cufftDoubleComplex *data;   // 显存数据指针
    cufftDoubleComplex *FFT1D_Bufcalib; // fft结果
    cufftDoubleComplex *data_cpu;
    cufftDoubleComplex *FFT1D_Bufcalib_cpu; 

    data_cpu = (cufftDoubleComplex *)malloc(sizeof(cufftDoubleComplex) * numPT * BATCH);
    FFT1D_Bufcalib_cpu = (cufftDoubleComplex *)malloc(sizeof(cufftDoubleComplex) * nfft_r * BATCH);

    CHECK(cudaMalloc((void**)&data, sizeof(cufftDoubleComplex) * numPT * BATCH));
    CHECK(cudaMalloc((void**)&FFT1D_Bufcalib, sizeof(cufftDoubleComplex) * nfft_r * BATCH));
    CHECK(cufftPlan1d(&plan, nfft_r, CUFFT_Z2Z, BATCH));

    // 输入数据
	for(int i = 0; i < numPT; i++) 
	{
		data_cpu[i].x = ADCBuf[i][0][0];    // 取第一个通道 第一个Chirp
		data_cpu[i].y = 0;
	}

    // 数据传输cpu->gpu
    CHECK(cudaMemcpy(data, data_cpu, sizeof(cufftDoubleComplex) * numPT * BATCH, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());

    CHECK(cufftExecZ2Z(plan, data, FFT1D_Bufcalib, CUFFT_FORWARD));

    // 数据传输gpu->cpu
    CHECK(cudaMemcpy(FFT1D_Bufcalib_cpu, FFT1D_Bufcalib, sizeof(cufftDoubleComplex) * nfft_r * BATCH, cudaMemcpyDeviceToHost));
	CHECK(cudaDeviceSynchronize());

    // 观察正半轴的数据
    double mo[nfft_r/2];
    double Ranlabel[nfft_r/2];

    for (int i = 0; i < nfft_r/2; i++) 
	{
		// printf("%f  %f\n", data_cpu[i].x, data_cpu[i].y);
		mo[i] = sqrt(FFT1D_Bufcalib_cpu[i].x * FFT1D_Bufcalib_cpu[i].x + FFT1D_Bufcalib_cpu[i].y * FFT1D_Bufcalib_cpu[i].y)*aaa;
		Ranlabel[i] = R_res * i / nfft_r * numPT ;
		// printf("%.15f  %.15f\n", Ranlabel[i], mo[i]);
	}

    cufftDestroy(plan); // 释放 GPU 资源
    
    cudaFree(data);
    cudaFree(data_cpu);
    cudaFree(FFT1D_Bufcalib);
    cudaFree(FFT1D_Bufcalib_cpu);

    cudaDeviceReset();

    // 去除近点干扰
    for(int i = 0; i < 3; i++)  mo[i] = 0;

    // 距离维搜索
    int RtarId = maxSearch(mo, nfft_r/2);
    double Rtar = Ranlabel[RtarId];
    // Rtar = 0.5399;  // debug
    printf("目标所在距离:%.4fm\n", Rtar);

    // ===========================近场亮温反演===========================   
    // // 计算补偿因子
    int azi_left = -30;
    int azi_right = 30;
    int pit_left = -30;
    int pit_right = 30;

    // int azi_left = -60;
    // int azi_right = 60;
    // int pit_left = -60;
    // int pit_right = 60;

    // 划分 方位x俯仰网格
    int Nay_net = round((azi_right-azi_left)*2);
    int Naz_net = round((pit_right-pit_left)*2);

    // 动态分配一维数组的内存，用于存储角度值
    double *Jd1_hori = (double *)malloc(Nay_net * sizeof(double));
    double *Jd1_pit = (double *)malloc(Naz_net * sizeof(double));
    double *Jd_hori = (double *)malloc(Nay_net * Naz_net * sizeof(double));
    double *Jd_pit = (double *)malloc(Nay_net * Naz_net * sizeof(double));

    // 初始化一维角度数组
    for (int i = 0; i < Nay_net; i++) {
        Jd1_hori[i] = azi_left + (azi_right - azi_left) / (double)(Nay_net - 1) * i;
    }
    for (int i = 0; i < Naz_net; i++) {
        Jd1_pit[i] = pit_left + (pit_right - pit_left) / (double)(Naz_net - 1) * i;
    }
    // 初始化二维角度数组
    for (int i = 0; i < Naz_net; i++) {
        for (int j = 0; j < Nay_net; j++) {
            Jd_hori[i*Naz_net + j] = Jd1_hori[j];
        }
    }
    for (int i = 0; i < Nay_net; i++) {
        for (int j = 0; j < Naz_net; j++) {
            Jd_pit[i*Nay_net + j] = Jd1_pit[i];
        }
    }

    // 计算与所有网格之间的距离
    int Na = NumANT;   // 256

    // 定义收发天线的下标
    int *txnum = NULL;
    int *rxnum = NULL;
    int *blank_num = NULL;
	txnum = (int *)malloc(Na * sizeof(int));
	rxnum = (int *)malloc(Na * sizeof(int));
    blank_num = (int *)malloc(Na * sizeof(int));

    for(int i = 1; i <= Na; i++)
    {
        txnum[i-1] = ceil((double)i/4);
        rxnum[i-1] = i-(txnum[i-1]-1)*4;
        blank_num[i-1] = ceil((double)i/(tx_PerArray*rx_PerArray))-1;
    }

    begintime = clock();	//计时开始

    // CUDA 2D成像
    // ========================计算补偿因子========================
    cufftDoubleComplex *A_Comp = (cufftDoubleComplex *)malloc(Na * Naz_net * Nay_net * sizeof(cufftDoubleComplex)); // 一维分配
    if (A_Comp == NULL) {
        printf("malloc failed!\n");
        return -1;
    }

     // 分配设备内存
    int *d_txnum, *d_rxnum, *d_blank_num;
    double *d_Jd_hori, *d_Jd_pit;
    cufftDoubleComplex *d_A_Comp;
    double *d_txPos_full_x, *d_txPos_full_y, *d_rxPos_full_x, *d_rxPos_full_y;

    // 分配设备内存
    cudaMalloc(&d_txPos_full_x, numArray * sizeof(double));
    cudaMalloc(&d_txPos_full_y, numArray * sizeof(double));
    cudaMalloc(&d_rxPos_full_x, numArray * sizeof(double));
    cudaMalloc(&d_rxPos_full_y, numArray * sizeof(double));
    cudaMalloc(&d_txnum, Na * sizeof(int));
    cudaMalloc(&d_rxnum, Na * sizeof(int));
    cudaMalloc(&d_blank_num, Na * sizeof(int));
    cudaMalloc(&d_Jd_hori, Nay_net * Naz_net * sizeof(double));
    cudaMalloc(&d_Jd_pit, Naz_net * Nay_net * sizeof(double));
    cudaMalloc(&d_A_Comp, Naz_net * Nay_net * Na * sizeof(cufftDoubleComplex));

    // 将数据从主机复制到设备
    cudaMemcpy(d_txPos_full_x, txPos_full_x, numArray * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_txPos_full_y, txPos_full_y, numArray * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rxPos_full_x, rxPos_full_x, numArray * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rxPos_full_y, rxPos_full_y, numArray * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_txnum, txnum, Na * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rxnum, rxnum, Na * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blank_num, blank_num, Na * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Jd_hori, Jd_hori, Naz_net * Nay_net * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Jd_pit, Jd_pit, Naz_net * Nay_net * sizeof(double), cudaMemcpyHostToDevice);

    // 定义block和grid的尺寸
    dim3 dimGrid(Naz_net, Nay_net); // 每个grid的尺寸为Naz_net x Nay_net
    dim3 dimBlock(Na, 1); // block尺寸为Na x 1

    computeAComp<<<dimGrid, dimBlock>>>(d_A_Comp, d_txPos_full_x, d_txPos_full_y, d_rxPos_full_x, d_rxPos_full_y, \
                                        d_txnum, d_rxnum, d_blank_num, lambda, \
                                        d_Jd_hori, d_Jd_pit, Rtar, k0, Na, Naz_net, Nay_net, tx_PerArray);
    cudaDeviceSynchronize();

    // 将结果从设备复制回主机
    cudaMemcpy(A_Comp, d_A_Comp, Naz_net * Nay_net * Na * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    // printf("完成计算A_Comp\n");
    // cudaDeviceReset();

    // for(int i=0; i < 1; i++){
    //     for(int j=0; j < Naz_net; j++){
    //         printf("%.4f+%.4fi ",A_Comp[j+Naz_net*i].x, A_Comp[j+Naz_net*i].y);
    //     }
    //     printf("\n");
    // }

    // ========================计算回波的协方差矩阵========================
    cufftDoubleComplex *R_sigdata = (cufftDoubleComplex *)malloc(Na * Na * sizeof(cufftDoubleComplex)); // cufftDoubleComplex *R_sigdata[Na][Na] flattened

    int chirpwanted = 3 - 1;    // 取第3个chirp
    for(int i = 0; i < Na; i++)
    {
        for(int j = 0; j < Na; j++)
        {   
            R_sigdata[i * Na + j].x = 0;
            R_sigdata[i * Na + j].y = 0;
            // printf("i=%d,j=%d\n",i,j);
            for(int k = 0; k < numPT; k++)
            {   
                // 完成校准
                R_sigdata[i * Na + j].x += ADCBuf[k][chirpwanted][j] * \
                                            (overallCal[0][j] * overallCal[0][i] + overallCal[1][j] * overallCal[1][i]) * ADCBuf[k][chirpwanted][i];
                R_sigdata[i * Na + j].y += ADCBuf[k][chirpwanted][j] * \
                                            (overallCal[0][j] * overallCal[1][i] - overallCal[1][j] * overallCal[0][i]) * ADCBuf[k][chirpwanted][i]; 
            }
            // printf("%d  %d  %f\n",i,j, R_sigdata[i * Na + j].x);
        }
    }
    // printf("完成计算R_sigdata\n");

    // ========================计算亮温分布========================
    double *T_sence_mo = (double *)malloc(Naz_net * Nay_net * sizeof(double));  // 保存亮温反演结果
    cuDoubleComplex *d_R_sigdata = NULL;
    double *d_T_sence_mo = NULL;

    // Allocate device memory
    cudaMalloc(&d_R_sigdata, Na * Na * sizeof(cuDoubleComplex));
    cudaMalloc(&d_T_sence_mo, Naz_net * Nay_net * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(d_R_sigdata, R_sigdata, Na * Na * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemset(d_T_sence_mo, 0, Naz_net * Nay_net * sizeof(double));

    // Define block and grid dimensions
    dim3 dimGrid_2(Naz_net, Nay_net); // 每个grid的尺寸为Naz_net x Nay_net
    dim3 dimBlock_2(Na, 1); // block尺寸为Na x 1

    // 亮温反演
    calculateBrightness<<<dimGrid_2, dimBlock_2>>>(d_A_Comp, d_R_sigdata, d_T_sence_mo, Na, Naz_net, Nay_net);  // d_A_Comp之前计算保留
    cudaMemcpy(T_sence_mo, d_T_sence_mo, Naz_net * Nay_net * sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    endtime = clock();	//计时结束
    printf("===========Imaging Completed===========\n");
    printf("Running Time: %.4fs\n", (double)(endtime - begintime) / 1E6);

    // 成像结果保存csv
    char c = '\n'; //定义换行转义字符
    FILE *fp; //定义文件指针
    if ((fp = fopen("/home/nx/CUDA_RadarImaging/nvvp_workspace/Cuda3DImaging_Project/ImagingResult_TwoTargets.csv","w")) == NULL)
    {
        printf("the file can not open..");
        exit(0);
    }    //出错处理

    for(int i = 0; i < Naz_net; i++)
    {
        for(int j = 0; j < Nay_net; j++)
        {
            // printf("%d  %d\n",i,j);
            if((i == Naz_net-1) && (j == Nay_net-1))
            {
                fprintf(fp,"%f",T_sence_mo[i*Naz_net+j]); 
            }
            else
            {
                fprintf(fp,"%f,",T_sence_mo[i*Naz_net+j]); 
            }
        }
        fprintf(fp,"%c",c);  
    }
    fclose(fp);
    printf("成像结果写入完成\n");

    // ========================释放========================
    // Free device memory
    cudaFree(d_A_Comp);
    cudaFree(d_R_sigdata);
    cudaFree(d_T_sence_mo);

    cudaFree(d_txnum);
    cudaFree(d_rxnum);
    cudaFree(d_blank_num);
    cudaFree(d_Jd_hori);
    cudaFree(d_Jd_pit);
    cudaFree(d_A_Comp);

    cudaFree(d_txPos_full_x);
    cudaFree(d_txPos_full_y);
    cudaFree(d_rxPos_full_x);
    cudaFree(d_rxPos_full_y);

    cudaFree(R_sigdata);
    free(A_Comp);
    free(Jd_hori);
    free(Jd_pit);
    free(Jd1_hori);
    free(Jd1_pit);

    free(T_sence_mo);
    free(txnum);
    free(rxnum);
    free(blank_num);
    free(overallCal);
    free(txPos_full);
    free(rxPos_full);

    for (int i = 0; i < num_sample_real; i++) {  
        for (int j = 0; j < numChirp; j++) {  
            free(ADCBuf[i][j]);  
        }  
        free(ADCBuf[i]);  
    }  
    free(ADCBuf);  

    return 0;
}


// ===========================CUDA核函数===========================
// // v1.0 适用于block(1,1)
// // 亮温反演 CUDA 内核函数
// __global__ void calculateBrightness(
//     cuDoubleComplex *A_Comp, 
//     cuDoubleComplex *R_sigdata, 
//     double *T_sence_mo, 
//     int Na, 
//     int Naz_net, 
//     int Nay_net
// ) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.y;
//     int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    
//     cuDoubleComplex T_sence_left[256];
//     cuDoubleComplex T_sence;
//     int j = idy;
//     int i = idx; 

//     // 计算第一步乘结果
//     for (int k = 0; k < Na; k++)        // 中间结果索引k
//     {
//         T_sence_left[k].x = 0;
//         T_sence_left[k].y = 0;
//         for(int k2 = 0; k2 < Na; k2++)  // R_sigdata的列索引k2
//         {
//             T_sence_left[k].x += (A_Comp[k2*Naz_net*Nay_net + j*Naz_net + i].x * R_sigdata[k2*Na + k].x + \
//                                   A_Comp[k2*Naz_net*Nay_net + j*Naz_net + i].y * R_sigdata[k2*Na + k].y);     // [i,j]网格处
//             T_sence_left[k].y += (A_Comp[k2*Naz_net*Nay_net + j*Naz_net + i].x * R_sigdata[k2*Na + k].y - \
//                                   A_Comp[k2*Naz_net*Nay_net + j*Naz_net + i].y * R_sigdata[k2*Na + k].x);     // [i,j]网格处
//         }
//     }

//     // 计算网格亮温值
//     T_sence.x = 0;
//     T_sence.y = 0;
//     for (int k = 0; k < Na; k++)        // 中间结果索引k
//     {
//         T_sence.x += (T_sence_left[k].x * A_Comp[k*Naz_net*Nay_net + j*Naz_net + i].x - \
//                       T_sence_left[k].y * A_Comp[k*Naz_net*Nay_net + j*Naz_net + i].y);
//         T_sence.y += (T_sence_left[k].x * A_Comp[k*Naz_net*Nay_net + j*Naz_net + i].y + \
//                       T_sence_left[k].y * A_Comp[k*Naz_net*Nay_net + j*Naz_net + i].x);   

//         // printf("%d  %d  %f+(%f)i\n",i,j,T_sence.x,T_sence.y);
//     }
    
//     T_sence_mo[blockId] = sqrt(T_sence.x * T_sence.x + T_sence.y * T_sence.y); // 取模

// }

// v2.0 加速版本
__global__ void calculateBrightness(
    cuDoubleComplex *A_Comp, 
    cuDoubleComplex *R_sigdata, 
    double *T_sence_mo, 
    int Na, 
    int Naz_net, 
    int Nay_net
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 当前线程的 x 坐标
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // 当前线程的 y 坐标

    int j = idy; // 网格点 y 坐标
    int i = idx; // 网格点 x 坐标

    // 每个线程负责一个网格点的计算
    if (i >= Naz_net || j >= Nay_net) {
        return; // 超出网格范围，直接返回
    }

    // 每个线程初始化自己的 T_sence_left
    cuDoubleComplex T_sence_left[256];
    for (int k = 0; k < Na; k++) {
        T_sence_left[k] = {0.0, 0.0};
    }

    // 循环展开优化：内层循环展开
    for (int k = 0; k < Na; k++) {
        // 手动展开循环，每次处理 4 个元素
        for (int k2 = 0; k2 < Na; k2 += 4) {
            // 第 1 个元素
            T_sence_left[k].x += (A_Comp[k2 * Naz_net * Nay_net + j * Naz_net + i].x * R_sigdata[k2 * Na + k].x +
                                  A_Comp[k2 * Naz_net * Nay_net + j * Naz_net + i].y * R_sigdata[k2 * Na + k].y);
            T_sence_left[k].y += (A_Comp[k2 * Naz_net * Nay_net + j * Naz_net + i].x * R_sigdata[k2 * Na + k].y -
                                  A_Comp[k2 * Naz_net * Nay_net + j * Naz_net + i].y * R_sigdata[k2 * Na + k].x);

            // 第 2 个元素
            if (k2 + 1 < Na) {
                T_sence_left[k].x += (A_Comp[(k2 + 1) * Naz_net * Nay_net + j * Naz_net + i].x * R_sigdata[(k2 + 1) * Na + k].x +
                                      A_Comp[(k2 + 1) * Naz_net * Nay_net + j * Naz_net + i].y * R_sigdata[(k2 + 1) * Na + k].y);
                T_sence_left[k].y += (A_Comp[(k2 + 1) * Naz_net * Nay_net + j * Naz_net + i].x * R_sigdata[(k2 + 1) * Na + k].y -
                                      A_Comp[(k2 + 1) * Naz_net * Nay_net + j * Naz_net + i].y * R_sigdata[(k2 + 1) * Na + k].x);
            }

            // 第 3 个元素
            if (k2 + 2 < Na) {
                T_sence_left[k].x += (A_Comp[(k2 + 2) * Naz_net * Nay_net + j * Naz_net + i].x * R_sigdata[(k2 + 2) * Na + k].x +
                                      A_Comp[(k2 + 2) * Naz_net * Nay_net + j * Naz_net + i].y * R_sigdata[(k2 + 2) * Na + k].y);
                T_sence_left[k].y += (A_Comp[(k2 + 2) * Naz_net * Nay_net + j * Naz_net + i].x * R_sigdata[(k2 + 2) * Na + k].y -
                                      A_Comp[(k2 + 2) * Naz_net * Nay_net + j * Naz_net + i].y * R_sigdata[(k2 + 2) * Na + k].x);
            }

            // 第 4 个元素
            if (k2 + 3 < Na) {
                T_sence_left[k].x += (A_Comp[(k2 + 3) * Naz_net * Nay_net + j * Naz_net + i].x * R_sigdata[(k2 + 3) * Na + k].x +
                                      A_Comp[(k2 + 3) * Naz_net * Nay_net + j * Naz_net + i].y * R_sigdata[(k2 + 3) * Na + k].y);
                T_sence_left[k].y += (A_Comp[(k2 + 3) * Naz_net * Nay_net + j * Naz_net + i].x * R_sigdata[(k2 + 3) * Na + k].y -
                                      A_Comp[(k2 + 3) * Naz_net * Nay_net + j * Naz_net + i].y * R_sigdata[(k2 + 3) * Na + k].x);
            }
        }
    }

    // 计算网格亮温值
    cuDoubleComplex T_sence = {0.0, 0.0};
    for (int k = 0; k < Na; k++) {
        T_sence.x += (T_sence_left[k].x * A_Comp[k * Naz_net * Nay_net + j * Naz_net + i].x -
                      T_sence_left[k].y * A_Comp[k * Naz_net * Nay_net + j * Naz_net + i].y);
        T_sence.y += (T_sence_left[k].x * A_Comp[k * Naz_net * Nay_net + j * Naz_net + i].y +
                      T_sence_left[k].y * A_Comp[k * Naz_net * Nay_net + j * Naz_net + i].x);
    }

    // 计算模长并存储结果
    int result_index = j * Naz_net + i;
    T_sence_mo[result_index] = sqrt(T_sence.x * T_sence.x + T_sence.y * T_sence.y);
}

// 计算A_Comp
__global__ void computeAComp(
    cufftDoubleComplex *A_Comp, 
    double *txPos_full_x, 
    double *txPos_full_y, 
    double *rxPos_full_x, 
    double *rxPos_full_y, 
    int *txnum, 
    int *rxnum, 
    int *blank_num, 
    double lambda, 
    double *Jd_hori, 
    double *Jd_pit, 
    double Rtar, 
    double k0, 
    int Na, 
    int Naz_net, 
    int Nay_net, 
    int tx_PerArray)
{
    int row = threadIdx.x;
    int col = threadIdx.y;
    int gridIdx = blockIdx.x;
    int gridIdy = blockIdx.y; 

    double TxLoc_x, TxLoc_y, RxLoc_x, RxLoc_y;
    
    // 每个线程处理一个元素
    int blockId = gridIdy * Naz_net + gridIdx;
    // int threadId = row + blockId * blockDim.x;
    int aComp_idx = blockId + row * Naz_net * Nay_net;  // A_Comp的索引，有别于threadId的Na*Naz_net*Nay_net

    // 计算TxLoc和RxLoc
    TxLoc_x = txPos_full_x[txnum[row]-1] * (1.01*lambda/2);
    TxLoc_y = txPos_full_y[txnum[row]-1] * (1.01*lambda/2);
    RxLoc_x = rxPos_full_x[rxnum[row]+blank_num[row]*tx_PerArray-1] * (1.01*lambda/2);
    RxLoc_y = rxPos_full_y[rxnum[row]+blank_num[row]*tx_PerArray-1] * (1.01*lambda/2);
    
    // 计算R_ij
    double R_ij = sqrt(pow(Rtar,2) + \
                       pow(Rtar*tan(Jd_hori[blockId]/180*PI) - TxLoc_x, 2) + \
                       pow(Rtar*tan(Jd_pit[blockId]/180*PI) + TxLoc_y, 2)) + \
                  sqrt(pow(Rtar,2) + \
                       pow(Rtar*tan(Jd_hori[blockId]/180*PI) - RxLoc_x, 2) + \
                       pow(Rtar*tan(Jd_pit[blockId]/180*PI) + RxLoc_y, 2));

    // 计算A_Comp
    A_Comp[aComp_idx].x = cos(k0*R_ij)/R_ij/Na;
    A_Comp[aComp_idx].y =-sin(k0*R_ij)/R_ij/Na;

    // printf("row=%d col=%d gridIdx=%d gridIdy=%d aComp_idx=%d\n", row, col, gridIdx, gridIdy, aComp_idx);
}

// ===========================子函数===========================
// Uart解包
// 8路数据合并一起解
void DataPacUart(int *RecDataBuff, int ***ADCBuf, struct commonCfg commonCfgStruct)
{
    int numPT = commonCfgStruct.numSamp;
    int numChirp = commonCfgStruct.numChirp;
    int numANT = commonCfgStruct.numANT;

    int dataFac = 2048;
    int Head = 0;
    int len = numPT*numChirp*numANT*commonCfgStruct.numByte;
    int frameNum = floor(len /(len+Head))*8;    // 表示包数,8路
    int RecDataSize = 1048576*frameNum;

    // 计算长数据中每个元素所在Frame的编号,第一维度Frame
    int **FrameDataBuff = (int **)malloc(frameNum * sizeof(int *)); // double *FrameDataBuff[8];
	for (int i = 0; i < frameNum; i++)
	{
		// double FrameDataBuff[8][RecDataSize/2/8]
		FrameDataBuff[i] = (int *)malloc(RecDataSize/2/frameNum * sizeof(int));
	}

    for(int i = 0; i < frameNum; i++)  
    {    
        for(int j = 0; j < len/frameNum/2; j++)
        {   
            FrameDataBuff[i][j] = (*(RecDataBuff+i*RecDataSize/frameNum+2*j+1) << 8) + *(RecDataBuff+i*RecDataSize/frameNum+2*j);
            if(FrameDataBuff[i][j] > 32767)
            {
                FrameDataBuff[i][j] = FrameDataBuff[i][j] - 65536;
            }
        }
    }
    
    // 整合雷达采集数据: frameNum x (len/frameNum/2) --> numPT x numChirp x numANT
    int FrameDataBuff_temp;
    int ADCBufId_temp;
    for(int iFrame = 0; iFrame < frameNum; iFrame++)
    {
        ADCBufId_temp = iFrame*numANT/frameNum; // 计算转换后在ADCBuf天线维的索引偏移
        // ADCBufId_temp = 0;
        // printf("%d  ", ADCBufId_temp);
        
        for(int k = 0; k < numChirp; k++)
        {
            for(int i = 0; i < numANT/frameNum; i++) // numANT = 32*8
            {
                for(int j = 0; j < numPT; j++)
                {   
                    // RePacket DataCube
                    FrameDataBuff_temp = FrameDataBuff[iFrame][numPT*i+numPT*numANT/frameNum*k+j];
                    if((8 <= i) && (i <= 11))
                    {
                        ADCBuf[j][k][i-4+ADCBufId_temp] = FrameDataBuff_temp;
                    }
                    else if((16 <= i) && (i <= 19))
                    {
                        ADCBuf[j][k][i-8+ADCBufId_temp] = FrameDataBuff_temp;
                    }
                    else if((24 <= i) && (i <= 27))
                    {
                        ADCBuf[j][k][i-12+ADCBufId_temp] = FrameDataBuff_temp;
                    }
                    else if((4 <= i) && (i <= 7))
                    {
                        ADCBuf[j][k][i+12+ADCBufId_temp] = FrameDataBuff_temp;
                    }
                    else if((12 <= i) && (i <= 15))
                    {
                        ADCBuf[j][k][i+8+ADCBufId_temp] = FrameDataBuff_temp;
                    }
                    else if((20 <= i) && (i <= 23))
                    {
                        ADCBuf[j][k][i+4+ADCBufId_temp] = FrameDataBuff_temp;
                    }
                    else
                    {
                        ADCBuf[j][k][i+ADCBufId_temp] = FrameDataBuff_temp;
                    }
                }  
            }
        }
    }
    
    // 释放内存
    for (int i = 0; i < frameNum; i++)
	{
		if (FrameDataBuff[i] != NULL)
		{
			free(FrameDataBuff[i]);
			FrameDataBuff[i] = NULL;
		}
	}
	if (FrameDataBuff != NULL)
	{
		free(FrameDataBuff);
		FrameDataBuff = NULL;
	}

}

// 查找最大值
int maxSearch(double a[],int n)
{
	int maxnum,i;
	maxnum = 0;
	for(i = 0; i < n; i++)
	{
		if(a[i] > a[maxnum])
		maxnum = i;
	}
	return maxnum;  //返回最大值下标
}

