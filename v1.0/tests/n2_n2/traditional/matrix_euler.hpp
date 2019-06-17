#pragma once

#include <iostream>
#include <iomanip>
#include <Eigen/Dense>

#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <cvode/cvode_direct.h>
#include <sundials/sundials_types.h>

#include "angledata.hpp"
#include "constants.hpp"
#include "masses.hpp"

//#include "n2_n2/ai_pes_n2_n2.hpp"
//#include "n2_n2/ai_ids_n2_n2.hpp"

// Correspondence of variables:
//      q1,     q2,      q3,     q4
// Theta1(N2), Phi, Theta2(N2), R
class MatrixEuler
{
public:
    //explicit MatrixEuler( AI_PES_n2_n2 * pes_, AI_IDS_n2_n2 * ids_ )
    explicit MatrixEuler( ) 
    {
        //pes = pes_;
        //ids = ids_;
        init();
    }

    static void init();

    static double hamiltonian( N_Vector y );
    static double lagrangian( N_Vector y );

    static const double MU1;
    static const double MU2;
    static const double MU3;
    
    static int syst( realtype t, N_Vector y, N_Vector ydot, void * user_data );

    void hamilton_to_lagrange_variables( N_Vector y, N_Vector out );
    void exchange_angles_lagrange( N_Vector y, N_Vector out );
    void lagrange_to_hamilton_variables( N_Vector y, N_Vector out );

    int getDIM() const { return DIM; }
    double angular_momentum_length( N_Vector y );
    Eigen::Vector3d angular_momentum( N_Vector y );

    void transform_dipole( Eigen::Vector3d & dip, N_Vector y );

private:
    //static AI_PES_n2_n2 * pes;
    //static AI_IDS_n2_n2 * ids;

    const int DIM = 14;

    static const double MU1L1SQ;
    static const double MU2L2SQ;
	
    static void fillIT( const AngleData * q1, const AngleData * q2, const AngleData * q3, double q4_sq );
    static void fillIT_dq1( const AngleData * q1 );
    static void fillIT_dq2( const AngleData * q2, const AngleData * q3 );
    static void fillIT_dq3( const AngleData * q2, const AngleData * q3 );
    static void fillIT_dq4( double q4 );

    static void fill_a( const AngleData * q3 );
    static void fill_a_q3( const AngleData * q3 );
    static void fill_A( const AngleData * q2, const AngleData * q3 );
    static void fill_A_q2( const AngleData * q2, const AngleData * q3 );
    static void fill_A_q3( const AngleData * q2, const AngleData * q3 );

    static void fillV( const AngleData * theta, const AngleData * psi );
    static void fillW( const AngleData * theta, const AngleData * psi );
    static void fillW_dt( const AngleData * theta, const AngleData * psi );
    static void fillW_dpsi( const AngleData * theta, const AngleData * psi );

    void fillS( N_Vector y );
    static Eigen::Vector3d molecular_dipole;
    static Eigen::Matrix3d matrixS;

    static double temp;

    static Eigen::Vector3d pe_vector;
    static Eigen::Vector4d p_vector;

    static Eigen::Matrix3d inertia_tensor;
    static Eigen::Matrix3d inertia_tensor_inverse;
    static Eigen::Matrix3d inertia_tensor_dq1;
    static Eigen::Matrix3d inertia_tensor_dq2;
    static Eigen::Matrix3d inertia_tensor_dq3;
    static Eigen::Matrix3d inertia_tensor_dq4;

    static Eigen::Matrix4d a;
    static Eigen::Matrix4d a_q3;
    static Eigen::Matrix4d a_inverse;
    static Eigen::Matrix<double, 3, 4> A;
    static Eigen::Matrix<double, 4, 3> At;
    static Eigen::Matrix<double, 3, 4> A_q2;
    static Eigen::Matrix<double, 3, 4> A_q3;

    static Eigen::Matrix3d V;
    static Eigen::Matrix3d W;
    static Eigen::Matrix3d W_dt;
    static Eigen::Matrix3d W_dpsi;

    static Eigen::Matrix<double, 3, 4> t1;
    static Eigen::Matrix<double, 4, 3> t2;

    static Eigen::Matrix3d G11;
    static Eigen::Matrix4d G22;
    static Eigen::Matrix<double, 3, 4> G12;

    static Eigen::Matrix3d G11_dq;
    static Eigen::Matrix4d G22_dq;
    static Eigen::Matrix<double, 3, 4> G12_dq;

    static Eigen::Vector3d J_vector;
    static Eigen::Matrix<double, 1, 3> J_vector_t;

    static Eigen::Vector4d dH_dp;
    static Eigen::Vector3d dH_dpe;
    static Eigen::Vector3d dH_dJ;
};
