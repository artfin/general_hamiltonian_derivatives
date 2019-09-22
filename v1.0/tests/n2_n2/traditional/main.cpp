#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>

#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <cvode/cvode_direct.h>
#include <sundials/sundials_types.h>

#include "matrix_euler.hpp"

const int DIM = 14;

void stl_to_cvode( std::vector<double> const& v, N_Vector y )
{
    for ( size_t k = 0; k < v.size(); ++k ) {
        NV_Ith_S(y, k) = v[k];
    }
}

void print( std::string const& name, N_Vector y, const int N ) 
{
    std::cout << std::fixed << std::setprecision(16);
    std::cout << name << ":" << std::endl;

    for ( int k = 0; k < N; ++k ) {
        std::cout << NV_Ith_S(y, k) << std::endl;
    }
} 

int main()
{
    MatrixEuler matrixEuler;
    matrixEuler.init();

    N_Vector y = N_VNew_Serial( DIM );
    N_Vector ydot = N_VNew_Serial( DIM );
    
    std::vector<double> yv = { 15.0, -6.00393, 0.753412, -21.1339, 3.64236, 21.3157, 0.487732, 193.326, 0.979626, 7.99349, 0.915534, 2.5582, 0.4936, 7.9012 }; 
	
    stl_to_cvode( yv, y );

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    
    matrixEuler.syst( 0.0, y, ydot, nullptr ); 

	const int cycles = 1e6;
	for ( long k = 0; k < cycles; ++k ) {
        matrixEuler.syst( 0.0, y, ydot, nullptr ); 
    }

	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>( t2 - t1 ).count();
	std::cout << "Total duration: " << duration << " ns" << std::endl;
	std::cout << "Duration per cycle: " << duration / cycles << " ns" << std::endl;	
    
   // print( "ydot", ydot, DIM ); 

    N_VDestroy( y );
    N_VDestroy( ydot );

    return 0;
}

