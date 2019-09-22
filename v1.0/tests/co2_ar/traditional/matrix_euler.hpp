#pragma once

#include <iostream>
#include <cmath>
#include <Eigen/Dense>

#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <cvode/cvode_direct.h>
#include <sundials/sundials_types.h>

#include "masses.hpp"
//#include "ai_pes_co2ar.hpp" 
//#include "ai_ids_co2ar.hpp"

class MatrixEuler
{
public:
    static const double mu1;
    static const double mu2;
    static const double l;
    static const double l2;
    
	static void init( );
	static int syst( realtype t, N_Vector y, N_Vector ydot, void * user_data );

	static double hamiltonian( N_Vector y );

    void transform_dipole( Eigen::Vector3d & transformed, const double R,
                                                      const double Theta,
                                                      const double phi,
                                                      const double theta,
                                                      const double psi );

private:
//	static AI_PES_co2_ar AI_PES;
//  static AI_IDS_co2_ar AI_IDS;

    static double temp;

    static void fillIT( const double R2, const double sinT, const double cosT );
    static void fillITR( const double R );
    static void fillITT( const double sinT, const double cosT );

    static void fill_a();
    static void fill_A();

    static void fillW( const double sin_psi, const double cos_psi, const double sin_t, const double cos_t );
    static void fillW_dt( const double sin_t, const double cos_t, const double sin_psi, const double cos_psi );
    static void fillW_dpsi( const double sin_psi, const double cos_psi, const double sin_t, const double cos_t );

    static Eigen::Matrix3d inertia_tensor;
    static Eigen::Matrix3d inertia_tensor_inverse;
    static Eigen::Matrix3d inertia_tensor_dr; // derivative by R
    static Eigen::Matrix3d inertia_tensor_dT; // derivative by Theta

    static Eigen::Matrix2d a;
    static Eigen::Matrix2d a_inverse;
    static Eigen::Matrix<double, 3, 2> A;
    static Eigen::Matrix<double, 2, 3> At;

    static Eigen::Matrix3d W;
    static Eigen::Matrix3d W_dt;
    static Eigen::Matrix3d W_dpsi;

    // вспомогательные матрицы при расчете G11, G12, G22
    static Eigen::Matrix3d t1;
    static Eigen::Matrix2d t2;

    static Eigen::Matrix3d G11;
    static Eigen::Matrix2d G22;
    static Eigen::Matrix<double, 3, 2> G12;

    static Eigen::Matrix3d G11_dr; // derivative by R
    static Eigen::Matrix2d G22_dr; // derivative by R
    static Eigen::Matrix<double, 3, 2> G12_dr; // derivative by R

    static Eigen::Matrix3d G11_dT; // derivative by Theta
    static Eigen::Matrix2d G22_dT; // derivative by Theta
    static Eigen::Matrix<double, 3, 2> G12_dT; // derivative by Theta

    static Eigen::Vector3d pe_vector; // vector of Euler impulses (p_phi, p_theta, p_psi)
    static Eigen::Vector2d p_vector; // vector of general impulses (pR, pT)

    static Eigen::Vector2d dH_dp;
    static Eigen::Vector3d dH_dpe;

    static Eigen::Matrix3d S_matrix;
    static Eigen::Vector3d molecular_dipole;
};
