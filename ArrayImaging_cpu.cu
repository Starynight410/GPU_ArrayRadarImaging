/*****************************************************************************

    Project: 综合孔径阵列雷达成像_cpu 2024_05_16
    ================

    读据本地雷达采集数据,收发阵元位置,校准权重,输出2D成像结果

    实现方式
    ================

    通过CPU直接计算干涉反演结果

    Compiling the program
    ===================

    Type `make` to compile the program. Alternatively, type the following commands:

    nvcc -o ArrayImaging_cpu ArrayImaging_cpu.cu --ptxas-options=-v --use_fast_math -lcublas -lcufft
    ./ArrayImaging
   
****************************************************************************/
#include "cufft.h"
#include "readCSV.h"
#include "readRadarData.h"
#include <math.h>
#include <complex.h>
#include <time.h> 

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


// ===========================main===========================
int main()
{
    int begintime, endtime;

    // read binFile
    char filePath[] = "./第一组_泡沫垫高A/Target50cm0d0d_8MCU.bin";  // 8路采集合并, size=1048576*8
    // char filePath[] = "./第二组_双目标泡沫垫高A/TwoTargets50cm10d0d_8MCU.bin";  // 8路采集合并, size=1048576*8
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
    char filename_overallCal[] = "Target50m0d0d_overallCal.csv";
    char line[4096];
    double **overallCal;
    overallCal = readCSV(filename_overallCal, line, overallCal); // 实部:overallCal[0][i] 虚部:overallCal[1][i]
    
    // 获取阵元位置坐标
    char filename_txPos[] = "txPos_full.csv";
    char filename_rxPos[] = "rxPos_full.csv";

    // char line[1024];
    double **txPos_full;
    double **rxPos_full;
    txPos_full = readCSV(filename_txPos, line, txPos_full);
    rxPos_full = readCSV(filename_rxPos, line, rxPos_full);

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
    printf("目标所在距离:%.4fm\n", Rtar);

    // ===========================近场亮温反演===========================
    // 计算补偿因子
    int azi_left = -30;
    int azi_right = 30;
    int pit_left = -30;
    int pit_right = 30;
    // int azi_left = -10;
    // int azi_right = 10;
    // int pit_left = -10;
    // int pit_right = 10;


    // 划分 方位x俯仰网格
    int Nay_net = round((azi_right-azi_left)*2);
    int Naz_net = round((pit_right-pit_left)*2);

    double Jd1_hori[Nay_net];
    double Jd1_pit[Naz_net];

    for(int i = 0; i < Nay_net; i++) Jd1_hori[i] = azi_left + (azi_right - azi_left) / (double)(Nay_net - 1) * i;
    for(int i = 0; i < Naz_net; i++) Jd1_pit[i]  = pit_left + (pit_right - pit_left) / (double)(Naz_net - 1) * i;
    
    // 下面生成网格,读文件读入阵元位置坐标和编号
    double Jd_hori[Nay_net][Nay_net];
    double Jd_pit[Naz_net][Naz_net];

    for (int i = 0; i < Naz_net; i++)
    {
        for (int j = 0; j < Nay_net; j++)
        {
            Jd_hori[i][j] = Jd1_hori[j];
        }
    }
    for (int i = 0; i < Naz_net; i++)
    {
        for (int j = 0; j < Nay_net; j++)
        {
            Jd_pit[i][j] = Jd1_pit[i];
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

    cufftDoubleComplex **A_Comp = (cufftDoubleComplex **)malloc(Na * sizeof(cufftDoubleComplex *)); // cufftDoubleComplex *A_Comp[Na];
	if ((A_Comp == NULL) || (A_Comp == NULL))
	{
		printf("malloc failed!\n");
		return -1;
	}

	for (int i = 0; i < Na; i++)
	{
		// cufftDoubleComplex A_Comp[Na][Naz_net*Nay_net]
		A_Comp[i] = (cufftDoubleComplex *)malloc(Naz_net*Nay_net * sizeof(cufftDoubleComplex));
	}

    double R_ij;
    double TxLoc[2], RxLoc[2];
    // 计算补偿因子A_Comp: 方位 x 俯仰 x 阵元
    for(int ii = 0; ii < Na; ii++)
    {   
        TxLoc[0] = txPos_full[0][txnum[ii]-1] * (1.01*lambda/2);
        TxLoc[1] = txPos_full[1][txnum[ii]-1] * (1.01*lambda/2);
        RxLoc[0] = rxPos_full[0][rxnum[ii]+blank_num[ii]*tx_PerArray-1] * (1.01*lambda/2);
        RxLoc[1] = rxPos_full[1][rxnum[ii]+blank_num[ii]*tx_PerArray-1] * (1.01*lambda/2);

        // 以第一个接收阵元为参考点
        for(int kk = 0; kk < Nay_net; kk++)
        {
            for(int mm = 0; mm < Naz_net; mm++)
            {   
                R_ij =  sqrt(pow(Rtar, 2) + \
                             pow(Rtar*tan(Jd_hori[kk][mm]/180*PI) - TxLoc[0], 2) + \
                             pow(Rtar*tan(Jd_pit[kk][mm]/180*PI) + TxLoc[1], 2)) + \
                        sqrt(pow(Rtar, 2) + \
                             pow(Rtar*tan(Jd_hori[kk][mm]/180*PI) - RxLoc[0], 2) + \
                             pow(Rtar*tan(Jd_pit[kk][mm]/180*PI) + RxLoc[1], 2));

                // 这里对应原程序中的W_CBF
                A_Comp[ii][kk*Naz_net+mm].x = cos(k0*R_ij)/R_ij/Na;   // 分别存储实部和虚部
                A_Comp[ii][kk*Naz_net+mm].y =-sin(k0*R_ij)/R_ij/Na; 
            }
        }
    }
    printf("完成计算A_Comp\n");
    // endtime = clock();	//计时结束

    // 计算回波的协方差矩阵
    cufftDoubleComplex **R_sigdata = (cufftDoubleComplex **)malloc(Na * sizeof(cufftDoubleComplex *)); // cufftDoubleComplex *R_sigdata[Na];
    for (int i = 0; i < Na; i++)
	{
		// cufftDoubleComplex R_sigdata[Na][Na]
		R_sigdata[i] = (cufftDoubleComplex *)malloc(Na * sizeof(cufftDoubleComplex));
	}

    int chirpwanted = 3 - 1;    // 取第3个chirp
    for(int i = 0; i < Na; i++)
    {
        for(int j = 0; j < Na; j++)
        {   
            R_sigdata[i][j].x = 0;
            R_sigdata[i][j].y = 0;
            // printf("i=%d,j=%d\n",i,j);
            for(int k = 0; k < numPT; k++)
            {   
                // 完成校准
                R_sigdata[i][j].x += ADCBuf[k][chirpwanted][j] * \
                                    (overallCal[0][j] * overallCal[0][i] + overallCal[1][j] * overallCal[1][i]) * ADCBuf[k][chirpwanted][i];
                R_sigdata[i][j].y += ADCBuf[k][chirpwanted][j] * \
                                    (overallCal[0][j] * overallCal[1][i] - overallCal[1][j] * overallCal[0][i]) * ADCBuf[k][chirpwanted][i]; 
            }
            // printf("%d  %d  %f\n",i,j, R_sigdata[i][j].x);
        }
    }
    // printf("完成计算R_sigdata\n");

    // ========================计算亮温分布========================
    // 场景亮温变量初始化 
    cufftDoubleComplex *T_sence_left = (cufftDoubleComplex *)malloc(Na * sizeof(cufftDoubleComplex));   // 左乘的中间结果, 复数 1 x Na
    cufftDoubleComplex T_sence;
    double **T_sence_mo = (double **)malloc(Na * sizeof(double *)); // double *T_sence_mo[Naz_net];
    for (int i = 0; i < Na; i++)
	{
		// double T_sence_mo[Naz_net][Nay_net]
		T_sence_mo[i] = (double *)malloc(Nay_net * sizeof(double));
	}
    
    // 亮温反演
    for(int j = 0; j < Nay_net; j++)
    {
        for (int i = 0; i < Naz_net; i++)
        {   
            // printf("i=%d,j=%d\n",i,j);

            // 计算第一步乘结果
            for (int k = 0; k < Na; k++)        // 中间结果索引k
            {
                T_sence_left[k].x = 0;
                T_sence_left[k].y = 0;
                for(int k2 = 0; k2 < Na; k2++)  // R_sigdata的列索引k2
                {
                    T_sence_left[k].x += (A_Comp[k2][j*Naz_net+i].x * R_sigdata[k2][k].x + \
                                          A_Comp[k2][j*Naz_net+i].y * R_sigdata[k2][k].y);     // [i,j]网格处
                    T_sence_left[k].y += (A_Comp[k2][j*Naz_net+i].x * R_sigdata[k2][k].y - \
                                          A_Comp[k2][j*Naz_net+i].y * R_sigdata[k2][k].x);     // [i,j]网格处
                }
            }

            // 计算网格亮温值
            T_sence.x = 0;
            T_sence.y = 0;
            for (int k = 0; k < Na; k++)        // 中间结果索引k
            {
                T_sence.x += (T_sence_left[k].x * A_Comp[k][j*Naz_net+i].x - T_sence_left[k].y * A_Comp[k][j*Naz_net+i].y);
                T_sence.y += (T_sence_left[k].x * A_Comp[k][j*Naz_net+i].y + T_sence_left[k].y * A_Comp[k][j*Naz_net+i].x);   
                T_sence_mo[i][j] = sqrt(T_sence.x * T_sence.x + T_sence.y * T_sence.y); // 取模
                // printf("%d  %d  %f+(%f)i\n",i,j,T_sence.x,T_sence.y);
            }
        }
    }

    endtime = clock();	//计时结束
    printf("===========Imaging Completed===========\n");
    printf("Running Time: %.4fs\n", (double)(endtime - begintime) / 1E6);

    // 成像结果保存csv
    char c = '\n'; //定义换行转义字符
    FILE *fp; //定义文件指针
    if ((fp = fopen("ImagingResult_TwoTargets.csv","w")) == NULL)
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
                fprintf(fp,"%f",T_sence_mo[i][j]); 
            }
            else
            {
                fprintf(fp,"%f,",T_sence_mo[i][j]); 
            }
        }
        fprintf(fp,"%c",c);  
    }
    fclose(fp);
    printf("成像结果写入完成\n");

    // ========================释放========================
	if (T_sence_left != NULL)
	{
		free(T_sence_left);
		T_sence_left = NULL;
	}

    for (int i = 0; i < Naz_net; i++)
	{
		if (T_sence_mo[i] != NULL)
		{
			free(T_sence_mo[i]);
			T_sence_mo[i] = NULL;
		}
	}
	if (T_sence_mo != NULL)
	{
		free(T_sence_mo);
		T_sence_mo = NULL;
	}

    for (int i = 0; i < Na; i++)
	{
		if (R_sigdata[i] != NULL)
		{
			free(R_sigdata[i]);
			R_sigdata[i] = NULL;
		}
	}
	if (R_sigdata != NULL)
	{
		free(R_sigdata);
		R_sigdata = NULL;
	}

    for (int i = 0; i < Na; i++)
	{
		if (A_Comp[i] != NULL)
		{
			free(A_Comp[i]);
			A_Comp[i] = NULL;
		}
	}
	if (A_Comp != NULL)
	{
		free(A_Comp);
		A_Comp = NULL;
	}

    if (txnum != NULL)
	{
		free(txnum);
		txnum = NULL;
	}
    if (rxnum != NULL)
	{
		free(rxnum);
		rxnum = NULL;
	}
    if (blank_num != NULL)
	{
		free(blank_num);
		blank_num = NULL;
	}
    
    if (overallCal != NULL)
	{
		free(overallCal);
		overallCal = NULL;
	}
    if (txPos_full != NULL)
	{
		free(txPos_full);
		txPos_full = NULL;
	}
    if (rxPos_full != NULL)
	{
		free(rxPos_full);
		rxPos_full = NULL;
	}

    for (int i = 0; i < num_sample_real; i++) {  
        for (int j = 0; j < numChirp; j++) {  
            free(ADCBuf[i][j]);  
        }  
        free(ADCBuf[i]);  
    }  
    free(ADCBuf);  

    return 0;
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
