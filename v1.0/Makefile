.PHONY: all clean

CCXX := g++
ASSEMBLER := -S
CXXFLAGS := -Wall -Wextra

INCLUDE_PARAMETERS := -I/home/artfin/Desktop/repos/co2_ar/bound_hep_correlation/v2.0/ 
INCLUDE_SUNDIALS := -I/home/artfin/Desktop/libs/sundials-4.1.0/instdir/include/  
INCLUDE_EIGEN := -I/usr/local/include/eigen3/ 
INCLUDE_COMMON := -I/home/artfin/Desktop/repos/common_h

all: hamiltonian_derivatives.o

hamiltonian_derivatives.o: hamiltonian_derivatives.cpp
	$(CCXX) $(CXXFLAGS) $(ASSEMBLER) -c $< -o $@ $(INCLUDE_PARAMETERS) $(INCLUDE_SUNDIALS) $(INCLUDE_EIGEN) $(INCLUDE_COMMON)

clean: 
	rm -rf *.o
