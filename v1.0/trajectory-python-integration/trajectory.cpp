//
// Created by artfin on 18.06.19.
//

#include "trajectory.hpp"

const realtype Trajectory::reltol = 1.0e-12;

int Trajectory::check_flag(void *flagvalue, const char *funcname, int opt)
/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns a flag so check if
 *            flag >= 0
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer
 */
{
    int *errflag;

/* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if (opt == 0 && flagvalue == nullptr)
    {
        fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n", funcname);
        exit( 1 );
    }

/* Check if flag < 0 */
    else if (opt == 1)
    {
        errflag = (int *) flagvalue;
        if ( *errflag == CV_TOO_MUCH_WORK )
        {
            std::cout << "[ERROR STATUS] CV_TOO_MUCH_WORK" << std::endl << std::endl;
            return tags::TRAJECTORY_CUT_TAG;
        }

        if (*errflag < 0)
        {
            fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n", funcname, *errflag);
            return tags::TRAJECTORY_CUT_TAG;
        }
    }

/* Check if function returned NULL pointer - no memory allocated */
    else if (opt == 2 && flagvalue == nullptr)
    {
        fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n", funcname);
        return(1);
    }

    return(0);
}

void Trajectory::set_initial_conditions( N_Vector ic ) {
    for ( int k = 0; k < NEQ; ++k ) {
        NV_Ith_S(y, k) = NV_Ith_S(ic, k);
    }

    CVodeReInit(cvode_mem, t0, y);
}

void Trajectory::set_initial_conditions( std::vector<double> const& ic ) {
    assert(static_cast<int>(ic.size()) == NEQ );

    for ( int k = 0; k < NEQ; ++k ) {
        NV_Ith_S(y, k) = ic[k];
    }

    CVodeReInit(cvode_mem, t0, y);
}

void Trajectory::PrintFinalStats()
{
    long int nst, nfe, nsetups, nje, nfeLS, nni, ncfn, netf, nge;
    int retval;

    retval = CVodeGetNumSteps(cvode_mem, &nst);
    check_flag( &retval, "CVodeGetNumSteps", 1);
    retval = CVodeGetNumRhsEvals(cvode_mem, &nfe);
    check_flag(&retval, "CVodeGetNumRhsEvals", 1);
    retval = CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
    check_flag(&retval, "CVodeGetNumLinSolvSetups", 1);
    retval = CVodeGetNumErrTestFails(cvode_mem, &netf);
    check_flag(&retval, "CVodeGetNumErrTestFails", 1);
    retval = CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
    check_flag(&retval, "CVodeGetNumNonlinSolvIters", 1);
    retval = CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
    check_flag(&retval, "CVodeGetNumNonlinSolvConvFails", 1);

    retval = CVodeGetNumJacEvals(cvode_mem, &nje);
    check_flag(&retval, "CVodeGetNumJacEvals", 1);
    retval = CVodeGetNumLinRhsEvals(cvode_mem, &nfeLS);
    check_flag(&retval, "CVodeGetNumLinRhsEvals", 1);

    retval = CVodeGetNumGEvals(cvode_mem, &nge);
    check_flag(&retval, "CVodeGetNumGEvals", 1);

    printf("\nFinal Statistics:\n");
    printf("nst = %-6ld RHS evals  = %-6ld nsetups = %-6ld nfeLS = %-6ld nje = %ld\n",
           nst, nfe, nsetups, nfeLS, nje);
    printf("nni = %-6ld ncfn = %-6ld netf = %-6ld nge = %ld\n \n",
           nni, ncfn, netf, nge);
}


int Trajectory::make_step( double & tout, double & t )
{
	int flag = CVode( cvode_mem, tout, y, &t, CV_NORMAL );
    if ( flag == CV_SUCCESS ) {
        return 0;
    } else {
        return flag;
    }
}


void Trajectory::report_point( std::ostream & os, double t )
{
    PhasePoint p( y, NEQ );
    os << t << " " << p << std::endl;
}


int Trajectory::run( double tmax, double sampling_time )
{
	int status; 
	double t = 0.0;
	double tout = sampling_time; 

    for ( int step_counter = 0; ; step_counter++, tout += sampling_time )	
	{
        if ( t > tmax ) {
            break;
        }

        status = make_step( tout, t );
        if ( status ) {
            std::cout << "[CVODE_ERROR] status = " << status << std::endl;
            return status; 
        }

        report_point(std::cout, t);
	}

    return 0;
}


