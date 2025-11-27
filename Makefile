CXX := clang++
CXXFLAGS := -std=c++17 -O2 -Xpreprocessor -fopenmp
LDFLAGS := -lomp

PKG_CONFIG := pkg-config
OPENCV_CFLAGS := $(shell $(PKG_CONFIG) --cflags opencv4)
OPENCV_LIBS := $(shell $(PKG_CONFIG) --libs opencv4)

LIBOMP_PREFIX := $(shell brew --prefix libomp)

all: serial

serial: serial.cpp recon.cpp metrics.cpp
	$(CXX) $(CXXFLAGS) \
	    -I$(LIBOMP_PREFIX)/include \
	    serial.cpp recon.cpp metrics.cpp \
	    -o serial \
	    -L$(LIBOMP_PREFIX)/lib $(LDFLAGS) \
	    $(OPENCV_CFLAGS) $(OPENCV_LIBS)

clean:
	rm -f serial
