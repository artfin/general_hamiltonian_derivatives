#pragma once

#include <nvector/nvector_serial.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <math.h>

class PhasePoint
{
public:
    PhasePoint( N_Vector p, int dimension ) 
    {
        this->dimension = dimension;
        v.resize( dimension );
        copy_to(p, v);

        map_angles();
    }

    void copy_to( N_Vector nv, std::vector<double> & v)
    {
        for ( int k = 0; k < dimension; ++k ) {
            v[k] = NV_Ith_S(nv, k);
        }
    }

    double map_angle( double ang, double lim )
    {
        return fmod( lim + fmod(ang, lim), lim );
    }

    void map_angles()
    {
        std::vector<int> theta_indices = { 2, 6 };
        std::vector<int> phi_indices = { 4, 8 };

        for ( int i : theta_indices ) {
            v[i] = map_angle(v[i], M_PI);
        }
        
        for ( int i : phi_indices ) {
            v[i] = map_angle(v[i], 2.0 * M_PI);
        }
    }
    
    friend std::ostream& operator<<(std::ostream& os, const PhasePoint& p);    
    
    int dimension;
    std::vector<double> v;
};

