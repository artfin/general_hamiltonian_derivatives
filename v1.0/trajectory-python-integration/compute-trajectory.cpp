#include <iostream>
#include <chrono>
#include <iomanip>
#include <functional>
#include <cassert>

#include "hamiltonian-parameters.hpp"
#include "co2-ar-derivatives-object.hpp"
#include "trajectory.hpp"

int main()
{
    using namespace co2_ar;

    double tmax, sampling_time;
    std::cin >> tmax >> sampling_time;

    double d;
    std::vector<double> phase_space_vector; 

    while ( std::cin >> d ) {
        phase_space_vector.push_back( d );
    }
    assert( phase_space_vector.size() == TOTAL_SIZE );
    
    HamiltonianDerivatives<Q_SIZE> ham = create_derivatives_object();
    
    Trajectory trajectory( ham );
    trajectory.set_initial_conditions( phase_space_vector );

    trajectory.run( tmax, sampling_time ); 

    return 0;
}

