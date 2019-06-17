#include <iostream>
#include <chrono>
#include <iomanip>
#include <functional>

#include <Eigen/Dense>

constexpr int IT_SIZE = 3;
constexpr int Q_SIZE = 2;
constexpr int TOTAL_SIZE = 10;
#include "generated.hpp"

#include "matrix_wrapper.hpp"
#include "hamiltonian_derivatives.hpp"

#include "numderivative.hpp"

int main()
{
    std::cout << std::fixed << std::setprecision(16);

    HamiltonianDerivatives<Q_SIZE> ham;

    ham.set_IT_filler( fill_IT );
    ham.set_A_filler( fill_A );
    ham.set_a_filler( fill_a );

    std::vector< HamiltonianDerivatives<Q_SIZE>::IT_filler_type > IT_derivatives_fillers = {
        fill_IT_dR,
        fill_IT_dTheta
    };

    std::vector< HamiltonianDerivatives<Q_SIZE>::A_filler_type > A_derivatives_fillers = {
        fill_A_dR,
        fill_A_dTheta
    };
    
    std::vector< HamiltonianDerivatives<Q_SIZE>::a_filler_type > a_derivatives_fillers = {
        fill_a_dR,
        fill_a_dTheta
    };

    ham.set_IT_derivatives_fillers( IT_derivatives_fillers );
    ham.set_A_derivatives_fillers( A_derivatives_fillers );
    ham.set_a_derivatives_fillers( a_derivatives_fillers );
    
	Eigen::Matrix<double, TOTAL_SIZE, 1> y;
    Eigen::Matrix<double, TOTAL_SIZE, 1> ydot;
	y << 15.0, -6.00393, 0.753412, -21.1339, 3.64236, 21.3157, 0.487732, 193.326, 0.979626, 7.99349; 

    std::function<double(Eigen::Matrix<double, TOTAL_SIZE, 1>)> compute_hamiltonian = 
        [&]( Eigen::Matrix<double, TOTAL_SIZE, 1> v ) {
            return ham.compute_hamiltonian( v );
        };

    const double h = 1.0e-7;
    NumDerivative<TOTAL_SIZE, 3> nd( compute_hamiltonian, h );

    ham.compute_derivatives( y, ydot );

    double analytic, numeric, difference;
    
    std::cout << "Analytical \t\t Numerical \t\t Difference" << std::endl;
    for ( int k = 0; k < TOTAL_SIZE; ++k ) {
        analytic = ydot(k);
        
        if ( k % 2 == 0 ) {
            nd.setArgument( k + 1 );
            numeric = nd.compute( y ); 
        } else {
            nd.setArgument( k - 1 );
            numeric = -nd.compute( y ); 
        }
        
        difference = std::abs(analytic - numeric);
        std::cout << analytic << "\t" << numeric << "\t" << difference << std::endl;
    }

    /*
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	const int cycles = 1e7;
	for ( long k = 0; k < cycles; ++k ) {
        ham.compute_derivatives( y, ydot );
    }

	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>( t2 - t1 ).count();
	std::cout << "Total duration: " << duration << " ns" << std::endl;
	std::cout << "Duration per cycle: " << duration / cycles << " ns" << std::endl;	
    */

    return 0;
}
