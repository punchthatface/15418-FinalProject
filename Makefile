PKG_CONFIG  := pkg-config
  
OPENCV_CFLAGS := $(shell $(PKG_CONFIG) --cflags opencv)
OPENCV_LIBS   := $(shell $(PKG_CONFIG) --libs   opencv)

# NOTE: add -DUSE_CUDA_REFINEMENT here too
CXXFLAGS   := -O3 -std=c++17 -fopenmp -DUSE_CUDA_REFINEMENT
NVCCFLAGS  := -O3 -std=c++17 -DUSE_CUDA_REFINEMENT -Xcompiler -fopenmp

TARGET := serial

CPU_SRCS   := serial.cpp recon.cpp metrics.cpp
CPU_OBJS   := $(CPU_SRCS:.cpp=.o)
CUDA_SRCS  := recon_cuda.cu
CUDA_OBJS  := $(CUDA_SRCS:.cu=.o)

all: $(TARGET)

$(TARGET): $(CPU_OBJS) $(CUDA_OBJS)
        $(CXX) $(CXXFLAGS) \
                $(CPU_OBJS) $(CUDA_OBJS) \
                $(OPENCV_LIBS) \
                -lcudart \
                -o $(TARGET)

%.o: %.cpp
        $(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

%.o: %.cu
        $(NVCC) $(NVCCFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

clean:
        rm -f $(TARGET) $(CPU_OBJS) $(CUDA_OBJS)
