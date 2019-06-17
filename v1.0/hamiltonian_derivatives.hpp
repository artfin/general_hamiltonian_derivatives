#pragma once

#include <iostream>
#include <vector>
#include <functional>

#include <Eigen/Dense>

// _N == DOF
template <int _N>
class HamiltonianDerivatives
{
public:
    const static int _TOTAL_N = 2 * _N + 6;

    typedef std::function<void(Eigen::Ref<Eigen::Matrix<double, IT_SIZE, IT_SIZE>>,
                               std::vector<QData> const&)> IT_filler_type;
    typedef std::function<void(Eigen::Ref<Eigen::Matrix<double, IT_SIZE, _N>>,
                               std::vector<QData> const&)> A_filler_type;
    typedef std::function<void(Eigen::Ref<Eigen::Matrix<double, _N, _N>>,
                               std::vector<QData> const&)> a_filler_type;

    HamiltonianDerivatives( ) 
    {
        q.resize( _N );
        euler_angles.resize( 3 );
    
        W_derivatives.resize( 2 );
        W_derivatives[0] = Eigen::Matrix<double, 3, 3>::Zero(3, 3);
        W_derivatives[1] = Eigen::Matrix<double, 3, 3>::Zero(3, 3); 
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
	
    void fillW( std::vector<QData> const& euler_angles )
	{	
    	W(0, 0) = euler_angles[2].sin / euler_angles[1].sin;
    	W(0, 1) = euler_angles[2].cos;
    	W(0, 2) = - euler_angles[2].sin * euler_angles[1].cos / euler_angles[1].sin;

    	W(1, 0) = euler_angles[2].cos / euler_angles[1].sin;
    	W(1, 1) = - euler_angles[2].sin;
    	W(1, 2) = - euler_angles[2].cos * euler_angles[1].cos / euler_angles[1].sin; 

    	W(2, 2) = 1.0;
	}
	
    void fillW_derivatives( std::vector<QData> const& euler_angles )
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

    void initialize( Eigen::Matrix<double, _TOTAL_N, 1> const& y )
    {
        for ( int k = 0; k < _N; ++k ) {
			q[k] = QData( y(2 * k) );
        }
		
        for ( int k = 0; k < _N; ++k ) {
			p(k) = y(2 * k + 1);
        }
       
        euler_angles[0] = QData( y(2 * _N) );
        euler_angles[1] = QData( y(2 * _N + 2) );
        euler_angles[2] = QData( y(2 * _N + 4) );
		
        pe(0) = y(2 * _N + 1);
		pe(1) = y(2 * _N + 3);
		pe(2) = y(2 * _N + 5);
    }
    
    double compute_hamiltonian( Eigen::Matrix<double, _TOTAL_N, 1> const& y )
    {
        initialize( y );

        fillW( euler_angles );
		J = W * pe;

		IT.fill( q );
		A.fill( q );
		a.fill( q );
        
        G11 = (IT.get() - A.get() * ainv * A.get().transpose()).inverse();
    	G22 = (a.get() - A.get().transpose() * ITI * A.get()).inverse();
    	G12 = - G11 * A.get() * ainv;
            
        double temp = 0.0;
        temp += 0.5 * J.transpose() * G11 * J;
        temp += 0.5 * p.transpose() * G22 * p;
        temp += J.transpose() * G12 * p;

        return temp;
    }

    void compute_derivatives( Eigen::Matrix<double, _TOTAL_N, 1> const& y, Eigen::Matrix<double, _TOTAL_N, 1> & ydot )
	// format of input vector:
	// R, pR, sequence of (angle, momenta) for the first monomer, sequence of (angle, momenta) for the second monomer, 
	// phi, p_phi, theta, p_theta, psi, p_psi 
    {
        initialize( y );

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

            ydot(2 * k + 1) = - temp; //- dU_dq[k];
        }
		
        for ( int k = 0; k < 2; ++k )
		{
			double temp = 0.0;
			temp += pe.transpose() * W_derivatives[k].transpose() * tmp; 
			ydot(2 * _N + 3 + 2 * k) = - temp;
		}	
    }

private:
    std::vector<QData> q; // generalized coordinates
    Eigen::Matrix<double, _N, 1> p; // generalized momenta

    std::vector<QData> euler_angles;
    Eigen::Matrix<double, 3, 1> pe; 
    Eigen::Matrix<double, 3, 1> J;

    Eigen::Matrix<double, 3, 1> tmp;

    Eigen::Matrix<double, 3, 3> W;
    std::vector< Eigen::Matrix<double, 3, 3> > W_derivatives;

    Eigen::Matrix<double, 3, 3> ITI;
    Eigen::Matrix<double, _N, _N> ainv;

    Eigen::Matrix<double, 3, 3> G11;
    Eigen::Matrix<double, _N, _N> G22;
    Eigen::Matrix<double, 3, _N> G12;

    Eigen::Matrix<double, 3, 3> G11_dq;
    Eigen::Matrix<double, _N, _N> G22_dq;
    Eigen::Matrix<double, 3, _N> G12_dq;

    Eigen::Matrix<double, _N, 1> dH_dp;
    Eigen::Matrix<double, 3, 1> dH_dpe;

    Eigen::Matrix<double, 3, 3> IT_dq;
    Eigen::Matrix<double, 3, _N> A_dq;
    Eigen::Matrix<double, _N, _N> a_dq;

    MatrixWrapper<_N, _N> a; 
    MatrixWrapper<IT_SIZE, IT_SIZE> IT;
    MatrixWrapper<IT_SIZE, _N> A;

    std::vector< MatrixWrapper<IT_SIZE, IT_SIZE> > IT_derivatives;
    std::vector< MatrixWrapper<IT_SIZE, _N> > A_derivatives;
    std::vector< MatrixWrapper<_N, _N> > a_derivatives;
};


