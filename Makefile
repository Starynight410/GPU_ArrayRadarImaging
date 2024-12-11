NVCC = nvcc
TARGET = ArrayImaging_gpu
SOURCE = $(TARGET).cu
NVCCFLAGS = -o $(TARGET) $(SOURCE) --ptxas-options=-v --use_fast_math -lcudart -lcublas -lcufft

all: $(TARGET) ShowCUDAImagingResult

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS)

run_cuda:
	./$(TARGET)

run_python:
	python ShowCUDAImagingResult.py

clean:
	rm -f $(TARGET)

ShowCUDAImagingResult: run_cuda
	$(MAKE) run_python

.PHONY: all clean run_cuda run_python ShowCUDAImagingResult