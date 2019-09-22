#pragma once

#include <iostream>
#include <vector>
#include <functional>

#include <Eigen/Dense>

#include <nvector/nvector_serial.h>

#include "QData.hpp"
#include "matrix_wrapper.hpp"

// _N == DOF
template <int _N>
class HamiltonianDerivatives
{
public:
    const static int _TOTAL_N = 2 * _N + 6;
    int getDIM() const { return _TOTAL_N; }

    typedef std::function<void(Eigen::Ref<Eigen::Matrix<double, IT_SIZE, IT_SIZE>>,
                               std::vector<QData> const&)> IT_filler_type;
    typedef std::function<void(Eigen::Ref<Eigen::Matrix<double, IT_SIZE, _N>>,
                               std::vector<QData> const&)> A_filler_type;
    typedef std::function<void(Eigen::Ref<Eigen::Matrix<double, _N, _N>>,
                               std::vector<QData> const&)> a_filler_type;
    typedef std::function<double(std::vector<QData> const&)> potential_type;
    typedef std::function<void(std::vector<QData> const&, std::vector<double>&)> potential_derivatives_type;

    HamiltonianDerivatives( ) 
    {
        q.resize( _N );
        euler_angles.resize( 3 );
    
        W_derivatives.resize( 2 );
        W_derivatives[0] = Eigen::Matrix<double, 3, 3>::Zero(3, 3);
        W_derivatives[1] = Eigen::Matrix<double, 3, 3>::Zero(3, 3);

        dU_dq.resize( _N ); 
    }

    void set_IT_filler( IT_filler_type filler ) 
    {
        IT.set_filler( filler );
    }   

    void set_A_filler( A_filler_type filler )
    {
        A.set_filler( filler );
    }

    void set_a_filler( a_filler_type filler ) 
    {
        a.set_filler( filler );
    }

    MatrixWrapper<IT_SIZE, IT_SIZE> & get_IT_wrapper() { return IT; }

    void set_IT_derivatives_fillers( std::vector<IT_filler_type> const& fillers )
    {
        for ( auto const& filler : fillers )  { 
            IT_derivatives.emplace_back( filler ); 
        }
    }
    
    void set_A_derivatives_fillers( std::vector<A_filler_type> const& fillers )
    {
        for ( auto const& filler : fillers ) {
            A_derivatives.emplace_back( filler );
        }
    }

    void set_a_derivatives_fillers( std::vector<a_filler_type> const& fillers )
    {
        for ( auto const& filler : fillers ) {
            a_derivatives.emplace_back( filler );
        }
    }

    void set_potential( potential_type potential_, potential_derivatives_type derivatives_ )
    {
        POTENTIAL_SET = true;
        potential = potential_;
        derivatives = derivatives_;
    }

    static void fillW( std::vector<QData> const& euler_angles )
	{	
    	W(0, 0) = euler_angles[2].sin / euler_angles[1].sin;
    	W(0, 1) = euler_angles[2].cos;
    	W(0, 2) = - euler_angles[2].sin * euler_angles[1].cos / euler_angles[1].sin;

    	W(1, 0) = euler_angles[2].cos / euler_angles[1].sin;
    	W(1, 1) = - euler_angles[2].sin;
    	W(1, 2) = - euler_angles[2].cos * euler_angles[1].cos / euler_angles[1].sin; 

    	W(2, 2) = 1.0;
	}
	
    static void fillW_derivatives( std::vector<QData> const& euler_angles )
	{
    	W_derivatives[0](0, 0) = - euler_angles[2].sin * euler_angles[1].cos / euler_angles[1].sin_sq;
    	W_derivatives[0](0, 2) = euler_angles[2].sin / euler_angles[1].sin_sq;

    	W_derivatives[0](1, 0) = -euler_angles[2].cos * euler_angles[1].cos / euler_angles[1].sin_sq;
    	W_derivatives[0](1, 2) = euler_angles[2].cos / euler_angles[1].sin_sq;
    	
        W_derivatives[1](0, 0) = euler_angles[2].cos / euler_angles[1].sin;
    	W_derivatives[1](0, 1) = - euler_angles[2].sin;
    	W_derivatives[1](0, 2) = - euler_angles[2].cos * euler_angles[1].cos / euler_angles[1].sin;

    	W_derivatives[1](1, 0) = - euler_angles[2].sin / euler_angles[1].sin;
    	W_derivatives[1](1, 1) = - euler_angles[2].cos;
    	W_derivatives[1](1, 2) = euler_angles[2].sin * euler_angles[1].cos / euler_angles[1].sin;
	}

    static void initialize( Eigen::Matrix<double, _TOTAL_N, 1> const& y )
    {
        for ( int k = 0; k < _N; ++k ) {
			q[k] = QData( y(2 * k) );
        }
		
        for ( int k = 0; k < _N; ++k ) {
			p(k) = y(2 * k + 1);
        }

        for ( int k = 0; k < 3; ++k ) {
            euler_angles[k] = QData( y(2*(_N + k)) );
        }

        for ( int k = 0; k < 3; ++k ) {
            pe(k) = y(2*(_N + k) + 1);
        }
    }

    template <int Eigen_matrix_size>
    static void cp( N_Vector nv, Eigen::Matrix<double, Eigen_matrix_size, 1> & m )
    {
        for ( int k = 0; k < Eigen_matrix_size; ++k ) {
            m(k) = NV_Ith_S(nv, k);
        }
    } 

    template <int Eigen_matrix_size>
    static void cp( Eigen::Matrix<double, Eigen_matrix_size, 1> const& m, N_Vector nv )
    {
        for ( int k = 0; k < Eigen_matrix_size; ++k ) { 
            NV_Ith_S(nv, k) = m(k);
        }
    }

    template <int Eigen_matrix_size>
    static void cp( std::vector<double> const& v, Eigen::Matrix<double, Eigen_matrix_size, 1> & m ) 
    {
        for ( int k = 0; k < _TOTAL_N; ++k ) {
            m(k) = v[k];
        }
    } 

    template <int Eigen_matrix_size>
    static void cp( Eigen::Matrix<double, Eigen_matrix_size, 1> & m, std::vector<double> & v )
    {
        for ( int k = 0; k < Eigen_matrix_size; ++k ) {
            v[k] = m(k);
        }
    } 

    static double compute_hamiltonian( std::vector<double> const& y )
    {
        cp(y, ycp);
        return compute_hamiltonian( ycp );
    } 

    static double compute_hamiltonian( N_Vector y )
    {
        cp(y, ycp);
        return compute_hamiltonian( ycp );
    }

    static double compute_hamiltonian( Eigen::Matrix<double, _TOTAL_N, 1> const& y )
    {
        initialize( y );

        fillW( euler_angles );
		J = W * pe;

		IT.fill( q );
		A.fill( q );
		a.fill( q );
		
        ITI = IT.get().inverse();
        ainv = a.get().inverse();
        
        G11 = (IT.get() - A.get() * ainv * A.get().transpose()).inverse();
    	G22 = (a.get() - A.get().transpose() * ITI * A.get()).inverse();
    	G12 = - G11 * A.get() * ainv;
            
        double temp = 0.0;
        temp += 0.5 * J.transpose() * G11 * J;
        temp += 0.5 * p.transpose() * G22 * p;
        temp += J.transpose() * G12 * p;

        if ( POTENTIAL_SET ) {
            temp += potential( q );
        }

        return temp; 
    }
  
    static void compute_derivatives( std::vector<double> const& y, std::vector<double> & ydot )
    {
        cp( y, ycp );
        compute_derivatives( ycp, ydotcp );
        cp( ydotcp, ydot );
    }

    static int compute_derivatives( realtype t, N_Vector y, N_Vector ydot, void * user_data )
    {
        (void)(t);
        (void)(user_data);
            
        cp( y, ycp );
        compute_derivatives( ycp, ydotcp );
        cp( ydotcp, ydot );

        return 0;
    } 

    static void compute_derivatives( Eigen::Matrix<double, _TOTAL_N, 1> const& y, Eigen::Matrix<double, _TOTAL_N, 1> & ydot )
	// format of input vector:
	// R, pR, sequence of (angle, momenta) for the first monomer, sequence of (angle, momenta) for the second monomer, 
	// phi, p_phi, theta, p_theta, psi, p_psi 
    {
        initialize( y );

        //std::cout << "y: " << std::endl << y << std::endl;

		fillW( euler_angles );
		fillW_derivatives( euler_angles );

		J = W * pe;
	    //std::cout << "J: " << std::endl << J << std::endl;

		IT.fill( q );
		A.fill( q );
		a.fill( q );
		
        //std::cout << "A: " << std::endl << A.get() << std::endl;
        //std::cout << "a: " << std::endl << a.get() << std::endl;

        ITI = IT.get().inverse();
		ainv = a.get().inverse();
  
        //std::cout << "IT: " << std::endl << IT.get() << std::endl;
        //std::cout << "ITI: " << std::endl << ITI << std::endl;
        //std::cout << "ainv: " << std::endl << ainv << std::endl;

        // try to optimize?    
        G11 = (IT.get() - A.get() * ainv * A.get().transpose()).inverse();
    	G22 = (a.get() - A.get().transpose() * ITI * A.get()).inverse();
    	G12 = - G11 * A.get() * ainv;
		
        //std::cout << "G11: " << std::endl << G11 << std::endl;	
        //std::cout << "G12: " << std::endl << G12 << std::endl;	
        //std::cout << "G22: " << std::endl << G22 << std::endl;	
	
		dH_dp = G22 * p + G12.transpose() * J;
		for ( int k = 0; k < _N; ++k ) {
			ydot(2 * k) = dH_dp(k);
        }

        tmp = G11 * J + G12 * p;
    	dH_dpe = W.transpose() * tmp; 
		for ( int k = 0; k < 3; ++k ) {
			ydot(2 * _N + 2 * k) = dH_dpe(k);
        }    
  
        if ( POTENTIAL_SET ) {  
            derivatives( q, dU_dq );
        }

        for ( int k = 0; k < _N; ++k )
        {
            IT_derivatives[k].fill( q );
            A_derivatives[k].fill( q );
            a_derivatives[k].fill( q );

            //IT_dq = Eigen::Matrix<double, 3, 3>::Zero(3, 3);
            //A_dq = Eigen::Matrix<double, 3, 2>::Zero(3, 2);
            //a_dq = Eigen::Matrix<double, 2, 2>::Zero(2, 2);

            IT_dq = IT_derivatives[k].get();
            A_dq = A_derivatives[k].get();
            a_dq = a_derivatives[k].get();
           
            //std::cout << "IT_dq: " << std::endl << IT_dq << std::endl; 
            //std::cout << "a_dq: " << std::endl << a_dq << std::endl;
            //std::cout << "A_dq: " << std::endl << A_dq << std::endl;

            G11_dq = - G11 * (IT_dq - A_dq * ainv * A.get().transpose() + A.get() * ainv * a_dq * ainv * A.get().transpose() - A.get() * ainv * A_dq.transpose()) * G11;
            G22_dq = - G22 * (a_dq - A_dq.transpose() * ITI * A.get() + A.get().transpose() * ITI * IT_dq * ITI * A.get() - A.get().transpose() * ITI * A_dq) * G22; 
            G12_dq = - G11_dq * A.get() * ainv - G11 * A_dq * ainv + G11 * A.get() * ainv * a_dq * ainv;

            //std::cout << "G11_dq: " << std::endl << G11_dq << std::endl;
            //std::cout << "G12_dq: " << std::endl << G12_dq << std::endl;
            //std::cout << "G22_dq: " << std::endl << G22_dq << std::endl;

            double temp = 0.0;
            temp += 0.5 * J.transpose() * G11_dq * J;
            temp += 0.5 * p.transpose() * G22_dq * p;
            temp += J.transpose() * G12_dq * p;
            
            if ( POTENTIAL_SET ) {
                ydot(2 * k + 1) = - temp - dU_dq[k];
            } else {
                ydot(2 * k + 1) = - temp;
            }
        }
		
        for ( int k = 0; k < 2; ++k )
		{
			double temp = 0.0;
			temp += pe.transpose() * W_derivatives[k].transpose() * tmp; 
			ydot(2 * _N + 3 + 2 * k) = - temp;
		}	
    }

private:
    static std::vector<QData> q; // generalized coordinates
    static Eigen::Matrix<double, _N, 1> p; // generalized momenta

    static Eigen::Matrix<double, 2*_N+6, 1> ycp;
    static Eigen::Matrix<double, 2*_N+6, 1> ydotcp;

    static std::vector<QData> euler_angles;
    static Eigen::Matrix<double, 3, 1> pe; 
    static Eigen::Matrix<double, 3, 1> J;

    static Eigen::Matrix<double, 3, 1> tmp;

    static Eigen::Matrix<double, 3, 3> W;
    static std::vector< Eigen::Matrix<double, 3, 3> > W_derivatives;

    static Eigen::Matrix<double, 3, 3> ITI;
    static Eigen::Matrix<double, _N, _N> ainv;

    static Eigen::Matrix<double, 3, 3> G11;
    static Eigen::Matrix<double, _N, _N> G22;
    static Eigen::Matrix<double, 3, _N> G12;

    static Eigen::Matrix<double, 3, 3> G11_dq;
    static Eigen::Matrix<double, _N, _N> G22_dq;
    static Eigen::Matrix<double, 3, _N> G12_dq;

    static Eigen::Matrix<double, _N, 1> dH_dp;
    static Eigen::Matrix<double, 3, 1> dH_dpe;

    static Eigen::Matrix<double, 3, 3> IT_dq;
    static Eigen::Matrix<double, 3, _N> A_dq;
    static Eigen::Matrix<double, _N, _N> a_dq;

    static MatrixWrapper<_N, _N> a; 
    static MatrixWrapper<IT_SIZE, IT_SIZE> IT;
    static MatrixWrapper<IT_SIZE, _N> A;

    static std::vector< MatrixWrapper<IT_SIZE, IT_SIZE> > IT_derivatives;
    static std::vector< MatrixWrapper<IT_SIZE, _N> > A_derivatives;
    static std::vector< MatrixWrapper<_N, _N> > a_derivatives;

    static bool POTENTIAL_SET;
    static potential_type potential;
    static potential_derivatives_type derivatives;
    static std::vector<double> dU_dq;
};

template <int _N>
std::function<double(std::vector<QData> const&)> HamiltonianDerivatives<_N>::potential;

template <int _N>
std::function<void(std::vector<QData> const&, std::vector<double>&)> HamiltonianDerivatives<_N>::derivatives;

template <int _N>
bool HamiltonianDerivatives<_N>::POTENTIAL_SET = false;

template<int _N>
std::vector<QData> HamiltonianDerivatives<_N>::q;

template<int _N>
Eigen::Matrix<double, _N, 1> HamiltonianDerivatives<_N>::p = Eigen::Matrix<double, _N, 1>::Zero(_N, 1);

template<int _N>
Eigen::Matrix<double, 2*_N+6, 1> HamiltonianDerivatives<_N>::ycp = Eigen::Matrix<double, 2*_N+6, 1>::Zero(2*_N+6, 1);

template<int _N>
Eigen::Matrix<double, 2*_N+6, 1> HamiltonianDerivatives<_N>::ydotcp = Eigen::Matrix<double, 2*_N+6, 1>::Zero(2*_N+6, 1);

template<int _N>
std::vector<QData> HamiltonianDerivatives<_N>::euler_angles;

template<int _N>
Eigen::Matrix<double, 3, 1> HamiltonianDerivatives<_N>::pe = Eigen::Matrix<double, 3, 1>::Zero(3, 1); 

template<int _N>
Eigen::Matrix<double, 3, 1> HamiltonianDerivatives<_N>::J = Eigen::Matrix<double, 3, 1>::Zero(3, 1);

template<int _N>
Eigen::Matrix<double, 3, 1> HamiltonianDerivatives<_N>::tmp = Eigen::Matrix<double, 3, 1>::Zero(3, 1);

template<int _N>
Eigen::Matrix<double, 3, 3> HamiltonianDerivatives<_N>::W = Eigen::Matrix<double, 3, 3>::Zero(3, 3);

template<int _N>
std::vector< Eigen::Matrix<double, 3, 3> > HamiltonianDerivatives<_N>::W_derivatives;

template<int _N>
Eigen::Matrix<double, 3, 3> HamiltonianDerivatives<_N>::ITI = Eigen::Matrix<double, 3, 3>::Zero(3, 3);

template<int _N>
Eigen::Matrix<double, _N, _N> HamiltonianDerivatives<_N>::ainv = Eigen::Matrix<double, _N, _N>::Zero(_N, _N);

template<int _N>
Eigen::Matrix<double, 3, 3> HamiltonianDerivatives<_N>::G11 = Eigen::Matrix<double, 3, 3>::Zero(3, 3);

template<int _N>
Eigen::Matrix<double, _N, _N> HamiltonianDerivatives<_N>::G22 = Eigen::Matrix<double, _N, _N>::Zero(_N, _N);

template<int _N>
Eigen::Matrix<double, 3, _N> HamiltonianDerivatives<_N>::G12 = Eigen::Matrix<double, 3, _N>::Zero(3, _N);

template<int _N>
Eigen::Matrix<double, 3, 3> HamiltonianDerivatives<_N>::G11_dq = Eigen::Matrix<double, 3, 3>::Zero(3, 3);

template<int _N>
Eigen::Matrix<double, _N, _N> HamiltonianDerivatives<_N>::G22_dq = Eigen::Matrix<double, _N, _N>::Zero(_N, _N);

template<int _N>
Eigen::Matrix<double, 3, _N> HamiltonianDerivatives<_N>::G12_dq = Eigen::Matrix<double, 3, _N>::Zero(3, _N);

template<int _N>
Eigen::Matrix<double, _N, 1> HamiltonianDerivatives<_N>::dH_dp = Eigen::Matrix<double, _N, 1>::Zero(_N, 1);

template<int _N>
Eigen::Matrix<double, 3, 1> HamiltonianDerivatives<_N>::dH_dpe= Eigen::Matrix<double, 3, 1>::Zero(3, 1);

template<int _N>
Eigen::Matrix<double, 3, 3> HamiltonianDerivatives<_N>::IT_dq = Eigen::Matrix<double, 3, 3>::Zero(3, 3);

template<int _N>
Eigen::Matrix<double, 3, _N> HamiltonianDerivatives<_N>::A_dq = Eigen::Matrix<double, 3, _N>::Zero(3, _N);

template<int _N>
Eigen::Matrix<double, _N, _N> HamiltonianDerivatives<_N>::a_dq = Eigen::Matrix<double, _N, _N>::Zero(_N, _N);

template<int _N>
MatrixWrapper<_N, _N> HamiltonianDerivatives<_N>::a; 

template<int _N>
MatrixWrapper<IT_SIZE, IT_SIZE> HamiltonianDerivatives<_N>::IT;

template<int _N>
MatrixWrapper<IT_SIZE, _N> HamiltonianDerivatives<_N>::A;

template<int _N>
std::vector< MatrixWrapper<IT_SIZE, IT_SIZE> > HamiltonianDerivatives<_N>::IT_derivatives;

template<int _N>
std::vector< MatrixWrapper<IT_SIZE, _N> > HamiltonianDerivatives<_N>::A_derivatives;

template<int _N>
std::vector< MatrixWrapper<_N, _N> > HamiltonianDerivatives<_N>::a_derivatives;

template<int _N>
std::vector<double> HamiltonianDerivatives<_N>::dU_dq;

