CXX := g++
CXXFLAGS := -std=c++17 -O2

PKG_CONFIG := pkg-config
OPENCV_CFLAGS := $(shell $(PKG_CONFIG) --cflags opencv4)
OPENCV_LIBS   := $(shell $(PKG_CONFIG) --libs opencv4)

# Default target
all: serial

serial: serial.cpp
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) serial.cpp -o serial $(OPENCV_LIBS)

clean:
	rm -f serial