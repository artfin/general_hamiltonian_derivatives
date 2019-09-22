#pragma once

#include <cmath>

struct QData 
{
public:
    QData( ) { }

    QData( const double value ) : value(value)
    {
        sq = value * value;

        sin = std::sin( value );
        cos = std::cos( value );

        sin_sq = sin * sin;
        cos_sq = cos * cos;
    }

    double value;
    double sq;
    double sin;
    double cos;
    double sin_sq;
    double cos_sq;
};


