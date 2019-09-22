#include <iostream>
#include <chrono>
#include <iomanip>
#include <functional>
#include <cassert>

constexpr int IT_SIZE = 3;
constexpr int Q_SIZE = 2;
constexpr int TOTAL_SIZE = 10;

#include "co2-ar-derivatives-object.hpp"

int main()
{
    using namespace co2_ar;

    std::cout << " # SYSTEM CO2-AR \n" << 
                 " # Compute Hamiltonian derivatives in the given point of the phase space\n";

    double d;
    std::vector<double> input, output;
    output.resize( TOTAL_SIZE );

    while ( std::cin >> d ) {
        input.push_back( d );
    }
    assert( input.size() == TOTAL_SIZE );
    
    HamiltonianDerivatives<Q_SIZE> ham = create_derivatives_object();
    ham.compute_derivatives( input, output ); 

    std::cout << std::fixed << std::setprecision(16);
    for ( double d : output ) {
        std::cout << d << std::endl; 
    }

    return 0;
}
