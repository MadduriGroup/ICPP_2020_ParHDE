.POSIX:
USE_MKL     = 0
CXX         = g++
CXXFLAGS    = -I. -O2 -Wall -pedantic -fopenmp
CC          = gcc
CFLAGS      = -std=c99 -O2 -Wall -pedantic -fopenmp #-qopenmp#-mkl=parallel#-qopt-report=5 -qopt-report-file=stdout -qopt-report-routine=HDE
LDLIBS      = -lm
GAPDIR      = ./gapbs
DBDIR	      = ./variants
MKLINCLUDE  = $(MKLROOT)/include

ifeq ($(USE_MKL), 1)
CXX 	      = icc
CC	        = icc
CXXFLAGS   += -qopenmp
CFLAGS     += -qopenmp
LDFLAGS    += -liomp5
CXXMKLFLAGS = -DMKL_ILP64 -I$(MKLINCLUDE) -DINTEL_MKL #-mkl=parallel#-qopt-report=5 -qopt-report-file=stdout -qopt-report-routine=HDE
LDXXLIBS    = -Wl,--start-group -L${MKLROOT}/lib/intel64 -Wl,--end-group -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
endif

# Add .exe on Windows
EXEEXT   = 

all: embed$(EXEEXT) draw$(EXEEXT) mtx2csr$(EXEEXT)

ifeq ($(USE_MKL), 1)
embed$(EXEEXT): spectralDrawing.cpp bfs$(EXEEXT) sssp$(EXEEXT) DrawingBase$(EXEEXT)
	$(CXX) $(CXXFLAGS) -I$(DBDIR) $(CXXMKLFLAGS) -std=c++11 -Wno-attributes spectralDrawing.cpp *.o -o embed$(EXEEXT) $(LDXXLIBS)
else
embed$(EXEEXT): spectralDrawing.cpp bfs$(EXEEXT) sssp$(EXEEXT) DrawingBase$(EXEEXT)
	$(CXX) $(CXXFLAGS) -I$(DBDIR) -std=c++11 -Wno-attributes spectralDrawing.cpp *.o -o embed$(EXEEXT) $(LDLIBS)
endif

mtx2csr$(EXEEXT): mtx2csr.cpp
	$(CXX) $(CXXFLAGS) mtx2csr.cpp -o mtx2csr$(EXEEXT)
draw$(EXEEXT): draw_graph.c
	$(CC) $(CFLAGS) draw_graph.c lodepng.c -o draw$(EXEEXT) $(LDLIBS)

bfs$(EXEEXT): $(GAPDIR)/bfs.cc
	$(CXX) -c $(CXXFLAGS) -std=c++11 $< -o bfs.o 
sssp$(EXEEXT): $(GAPDIR)/sssp.cc
	$(CXX) -c $(CXXFLAGS) -std=c++11 $< -o sssp.o 

ifeq ($(USE_MKL), 1)
DrawingBase$(EXEEXT): $(DBDIR)/*.cpp
	$(CXX) -c $(CXXFLAGS) $(CXXMKLFLAGS) -std=c++11 -I$(DBDIR) $^ $(LDXXLIBS)
else
DrawingBase$(EXEEXT): $(DBDIR)/*.cpp
	$(CXX) -c $(CXXFLAGS) -std=c++11 -I$(DBDIR) $^ $(LDLIBS)
endif


standalone$(EXEEXT): spectralDrawing.cpp
	$(CXX) $(CXXFLAGS) $(CXXMKLFLAGS) -std=c++11 -Wno-attributes standaloneSP.cpp -o standalone$(EXEEXT) $(LDXXLIBS)

clean:
	rm -f embed$(EXEEXT) mtx2csr$(EXEEXT) draw$(EXEEXT) *.optrpt standalone *.o

tags: *.c *.h
	ctags -R .
