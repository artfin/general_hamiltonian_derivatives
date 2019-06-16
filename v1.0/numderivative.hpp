#pragma once

#include <functional>

#include <Eigen/Dense>

template< int _Args >
class FunctionWrapper
{
public:
	FunctionWrapper( std::function<double(Eigen::Matrix<double, _Args, 1>)> f, unsigned int n,
                     Eigen::Matrix<double, _Args, 1> const& v ) : f(f), n(n), v(v) 
    { 
    }

	double operator()(double p) 
	{
		v(n) = p;
		return f(v);
	}

private:
	std::function<double(Eigen::Matrix<double, _Args, 1>)> f;
	unsigned int n;
    Eigen::Matrix<double, _Args, 1> v;
};

template< int _Args, int _Points >
class NumDerivative
{
public:
	NumDerivative( std::function<double(Eigen::Matrix<double, _Args, 1>)> f, const double h );
	void setArgument( const unsigned int n );
	double compute( Eigen::Matrix<double, _Args, 1> const& v );

private:
	std::function<double(Eigen::Matrix<double, _Args, 1>)> f;
	double h;
	unsigned int n;
};

template< int _Args >
class NumDerivative<_Args, 3>
{
public:
	NumDerivative( std::function<double(Eigen::Matrix<double, _Args, 1>)> f, const double h ) : 
        f(f), h(h) 
    { 
    }

	void setArgument( const unsigned int n ) { this->n = n; }
	
	double compute( Eigen::Matrix<double, _Args, 1> const& v )
	{
		FunctionWrapper<_Args> wrapper( f, n, v );

		double p = v[n];
		return (wrapper(p + h) - wrapper(p - h)) / (2.0 * h);	
	}

private:
	std::function<double(Eigen::Matrix<double, _Args, 1>)> f;
	double h;
	unsigned int n;
};

template< int _Args >
class NumDerivative<_Args, 5>
{
public:
	NumDerivative( std::function<double(Eigen::Matrix<double, _Args, 1>)> f, const double h ) : 
        f(f), h(h) 
    { 
    }

	void setArgument( const unsigned int n ) { this->n = n; }
	
	double compute( Eigen::Matrix<double, _Args, 1> const& v )
	{
		FunctionWrapper<_Args> wrapper( f, n, v );

		double p = v[n];
		return (wrapper(p - 2.0*h) - 8.0*wrapper(p - h) + 8.0*wrapper(p + h) - wrapper(p + 2.0*h)) / (12.0 * h); 
	}

private:
	std::function<double(Eigen::Matrix<double, _Args, 1>)> f;
	double h;
	unsigned int n;
};

template< int _Args >
class NumDerivative<_Args, 7>
{
public:
	NumDerivative( std::function<double(Eigen::Matrix<double, _Args, 1>)> f, const double h ) : 
        f(f), h(h) 
    { 
    }

	void setArgument( const unsigned int n ) { this->n = n; }
	
	double compute( Eigen::Matrix<double, _Args, 1> const& v )
	{
		FunctionWrapper<_Args> wrapper( f, n, v );
		double p = v[n];
		return (-wrapper(p - 3.0*h) + 9.0*wrapper(p - 2.0 * h) - 45.0*wrapper(p - h) + 45.0*wrapper(p + h) - 9.0*wrapper(p + 2.0*h) + wrapper(p + 3.0*h)) / (60.0 * h); 
	}

private:
	std::function<double(Eigen::Matrix<double, _Args, 1>)> f;
	double h;
	unsigned int n;
};

