#include "phase_point.hpp"

std::ostream& operator<<(std::ostream& os, const PhasePoint& p)
{
    os << std::fixed << std::setprecision(16);
    for ( int k = 0; k < p.dimension; ++k ) {
        os << p.v[k] << " ";
    }

    return os;
} 


