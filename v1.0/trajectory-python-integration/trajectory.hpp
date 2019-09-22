//
// Created by artfin on 21.03.19.
//

#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <cvode/cvode_direct.h>
#include <sundials/sundials_types.h>

#include "hamiltonian-parameters.hpp"
#include "hamiltonian_derivatives.hpp"
#include "phase_point.hpp"

#include "tags.hpp"

class Trajectory {

public:
    Trajectory( HamiltonianDerivatives<Q_SIZE> & ham ) : ham(ham) 
    {
        NEQ = ham.getDIM();
        ic = std::vector<double>( NEQ );

        y = N_VNew_Serial( NEQ );
        if ( check_flag( (void*) y, "N_VNew_Serial", 0) )
            throw std::runtime_error( "CHECK_FLAG N_Vnew_serial" );

        ydot = N_VNew_Serial( NEQ );
        if ( check_flag( (void*) ydot, "N_VNew_Serial", 0) )
            throw std::runtime_error( "CHECK_FLAG N_Vnew_serial" );

        abstol = N_VNew_Serial( NEQ );
        if ( check_flag( (void*) abstol, "N_VNew_Serial", 0) )
            throw std::runtime_error( "CHECK_FLAG N_VNew_Serial" );

        for ( int k = 0; k < NEQ; ++k )
            NV_Ith_S( abstol, k ) = reltol;

        // call CVodeCreate to create the solver memory and specify the
        // Backward Differentiation Formula and the use of a Newton iteration
        cvode_mem = CVodeCreate( CV_BDF );
        if ( check_flag( (void*) cvode_mem, "CVodeCreate", 0) )
            throw std::runtime_error( "CHECK_FLAG CVodeCreate" );
       
        int flag;
        // call CVodeInit to initialize the integrator memory and specify the
        // user's right hand side (rhs) function in y' = f(t, y), the initial
        // time t0, and the initial dependent variable vector 'y'
        flag = CVodeInit( cvode_mem, ham.compute_derivatives, t0, y);
        if( check_flag( &flag, "CVodeInit", 1) )
            throw std::runtime_error( "CHECK_FLAG CVodeInit" );

        // call CVodeSVtolerances to specify the scalar relative tolerance
        // and vector absolute tolerances
        flag = CVodeSVtolerances( cvode_mem, reltol, abstol );
        if ( check_flag( &flag, "CVodeSVtolerances", 1) )
            throw std::runtime_error( "CHECK_FLAG CVodeSVtolerances" );

        // Create dense SUNMatrix for use in linear solves
        A = SUNDenseMatrix( NEQ, NEQ );
        if ( check_flag( (void*) A, "SUNDenseMatrix", 0) )
            throw std::runtime_error( "CHECK_FLAG SUNDenseMatrix" );

        // create dense SUNLinearSolver object for use by CVode
        LS = SUNDenseLinearSolver( y, A );
        if ( check_flag( (void*) LS, "SUNDenseLinearSolver", 0) )
            throw std::runtime_error( "CHECK_FLAG SUNDenseLinearSolver" );

        // Call CVDlsSetLinearSolver to attach the matrix and linear solver to CVode
        flag = CVDlsSetLinearSolver( cvode_mem, LS, A );
        if ( check_flag( &flag, "CVDlsSetLinearSolver", 1) )
            throw std::runtime_error( "CHECK_FLAG CVDlsSetLinearSolver" );

        // number of internal steps before tout; default= 500
        CVodeSetMaxNumSteps( cvode_mem, static_cast<int>(5e5) );
    }

    ~Trajectory( )
    {
        N_VDestroy( y );
        N_VDestroy( ydot );
        N_VDestroy( abstol );

        CVodeFree( &cvode_mem );

        SUNLinSolFree( LS );
        SUNMatDestroy( A );
    }

    int check_flag(void *flagvalue, const char *funcname, int opt);

    void set_initial_conditions( std::vector<double> const& ic );
    void set_initial_conditions( N_Vector ic );
    static double get_relative_precision() { return static_cast<double>(reltol); }

    int make_step( double & tout, double & t );
    void report_point( std::ostream & os, double t );
    int run( double tmax, double sampling_time = 200.0 );

    void PrintFinalStats();

    N_Vector get_y() {
        return y;
    }

    double map_ang( double ang, double lim );

    std::vector<double> ic;

private:
    HamiltonianDerivatives<Q_SIZE> & ham;

    int NEQ;
    double sampling_time;

    int cut_counter = 0;

    N_Vector y = nullptr;
    N_Vector ydot = nullptr;
    N_Vector abstol = nullptr;
    static const realtype reltol;
    SUNMatrix A = nullptr;
    SUNLinearSolver LS = nullptr;

    void * cvode_mem;

    realtype t0 = 0.0;
    realtype t  = 0.0;
    realtype tout = 0.0;
};

