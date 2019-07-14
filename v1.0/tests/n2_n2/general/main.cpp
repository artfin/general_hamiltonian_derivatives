#include <iostream>
#include <chrono>
#include <iomanip>
#include <functional>

#include <Eigen/Dense>

constexpr int IT_SIZE = 3;
constexpr int Q_SIZE = 4;
constexpr int TOTAL_SIZE = 2 * Q_SIZE + 6;
#include "generated.hpp"

#include "QData.hpp"
#include "matrix_wrapper.hpp"
#include "hamiltonian_derivatives.hpp"
#include "numderivative.hpp"

#include "n2_n2/ai_pes_n2_n2.hpp"

std::vector<double> reorder( std::vector<double> const& v, std::vector<size_t> const &order )  
{  
    std::vector<double> result;   
    result.resize( v.size() ); 

    for ( size_t k = 0; k < order.size(); ++k ) {
        result[k] = v[ order[k] ];
    }

    return result; 
}

int main()
{
    std::cout << std::fixed << std::setprecision(16);

    HamiltonianDerivatives<Q_SIZE> ham;

    ham.set_IT_filler( fill_IT ); 
    ham.set_A_filler( fill_A );
    ham.set_a_filler( fill_a );
    
    std::vector< HamiltonianDerivatives<Q_SIZE>::IT_filler_type > IT_derivatives_fillers = {
        fill_IT_dq1,
        fill_IT_dq2,
        fill_IT_dq3,
        fill_IT_dq4
    };

    std::vector< HamiltonianDerivatives<Q_SIZE>::A_filler_type > A_derivatives_fillers = {
        fill_A_dq1,
        fill_A_dq2,
        fill_A_dq3,
        fill_A_dq4
    };
    
    std::vector< HamiltonianDerivatives<Q_SIZE>::a_filler_type > a_derivatives_fillers = {
        fill_a_dq1,
        fill_a_dq2,
        fill_a_dq3,
        fill_a_dq4
    };
    
    ham.set_IT_derivatives_fillers( IT_derivatives_fillers );
    ham.set_A_derivatives_fillers( A_derivatives_fillers );
    ham.set_a_derivatives_fillers( a_derivatives_fillers );


    AI_PES_n2_n2 pes;
    ham.set_potential( pes.pes_q, pes.pes_derivatives );

    Eigen::Matrix<double, TOTAL_SIZE, 1> y;
    Eigen::Matrix<double, TOTAL_SIZE, 1> ydot;
	y << 0.753412, -21.1339, 3.64236, 21.3157, 0.487732, 193.326, 15.0, -6.00393, 0.979626, 7.99349, 0.915534, 2.5582, 0.4936, 7.9012; 
  
   /* 
    std::vector<QData> q = {
        QData(0.753412),
        QData(3.64236),
        QData(0.487732),
        QData(15.0)
    };
    */
   
    std::function<double(Eigen::Matrix<double, TOTAL_SIZE, 1>)> compute_hamiltonian = 
        [&]( Eigen::Matrix<double, TOTAL_SIZE, 1> v ) {
            return ham.compute_hamiltonian( v );
        };

    const double h = 1.0e-3;
    NumDerivative<TOTAL_SIZE, 5> nd( compute_hamiltonian, h );

    ham.compute_derivatives( y, ydot );
    
    std::vector<double> ydot_v( &ydot[0], ydot.data() + ydot.rows() );
    std::vector<size_t> ordering = { 6, 7, 0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13 }; 
    std::vector<double> analytic = reorder( ydot_v, ordering );

    std::vector<double> numeric_;
    for ( int k = 0; k < TOTAL_SIZE; ++k ) {
        double num;

        if ( k % 2 == 0 ) {
            nd.setArgument( k + 1 );
            num = nd.compute( y ); 
        } else {
            nd.setArgument( k - 1 );
            num = -nd.compute( y ); 
        }

        numeric_.push_back( num );
    }
   
    std::vector<double> numeric = reorder( numeric_, ordering );

    double difference;
   
    std::cout << "Analytical \t\t Numerical \t\t Difference" << std::endl;
    for ( int k = 0; k < TOTAL_SIZE; ++k ) {
        difference = std::abs(analytic[k] - numeric[k]);
        std::cout << analytic[k] << "\t" << numeric[k] << "\t" << difference << std::endl;
    }

    /*
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	const int cycles = 1e5;
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
