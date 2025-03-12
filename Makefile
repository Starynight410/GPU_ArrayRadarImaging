NVCC = nvcc
# TARGET = SampleImaging_v1
TARGET = ArrayImaging_gpu
SOURCE = $(TARGET).cu
NVCCFLAGS = -o $(TARGET) $(SOURCE) --ptxas-options=-v --use_fast_math -lcudart -lcublas -lcufft
PYTHON_SCRIPT_PATH = /home/nx/CUDA_RadarImaging/nvvp_workspace/Cuda3DImaging_Project/ShowCUDAImagingResult.py

.PHONY: all clean run_cuda run_python

# 默认目标：执行清理、编译、运行 CUDA 和运行 Python 脚本
all: clean compile run_cuda run_python

# 清理已编译的文件
clean:
	rm -f $(TARGET)

# 编译 CUDA 程序
compile: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS)

# 运行编译后的 CUDA 程序
run_cuda: $(TARGET)
	./$(TARGET)

# 运行 Python 脚本
run_python:
	python3 $(PYTHON_SCRIPT_PATH)