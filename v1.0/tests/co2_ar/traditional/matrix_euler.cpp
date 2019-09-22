#include "matrix_euler.hpp"

//AI_PES_co2_ar MatrixEuler::AI_PES = AI_PES_co2_ar();
//AI_IDS_co2_ar MatrixEuler::AI_IDS = AI_IDS_co2_ar();

const double MatrixEuler::mu1 = masses::O16_AMU / 2.0; 
const double MatrixEuler::mu2 = masses::CO2_AMU * masses::AR40_AMU / (masses::CO2_AMU + masses::AR40_AMU);
const double MatrixEuler::l = lengths::L_CO2_ALU;
const double MatrixEuler::l2 = MatrixEuler::l * MatrixEuler::l;
double MatrixEuler::temp = 0.0;

Eigen::Matrix3d MatrixEuler::inertia_tensor = Eigen::Matrix3d::Zero(3, 3);
Eigen::Matrix3d MatrixEuler::inertia_tensor_inverse = Eigen::Matrix3d::Zero(3, 3);
Eigen::Matrix3d MatrixEuler::inertia_tensor_dr = Eigen::Matrix3d::Zero(3, 3);
Eigen::Matrix3d MatrixEuler::inertia_tensor_dT = Eigen::Matrix3d::Zero(3, 3);

Eigen::Matrix2d MatrixEuler::a = Eigen::Matrix2d::Zero(2, 2);
Eigen::Matrix2d MatrixEuler::a_inverse = Eigen::Matrix2d::Zero(2, 2);
Eigen::Matrix<double, 3, 2> MatrixEuler::A = Eigen::Matrix<double, 3, 2>::Zero(3, 2);
Eigen::Matrix<double, 2, 3> MatrixEuler::At = Eigen::Matrix<double, 2, 3>::Zero(2, 3);

Eigen::Matrix3d MatrixEuler::W = Eigen::Matrix3d::Zero(3, 3);
Eigen::Matrix3d MatrixEuler::W_dt = Eigen::Matrix3d::Zero(3, 3);
Eigen::Matrix3d MatrixEuler::W_dpsi = Eigen::Matrix3d::Zero(3, 3);

Eigen::Matrix3d MatrixEuler::t1 = Eigen::Matrix3d::Zero(3, 3);
Eigen::Matrix2d MatrixEuler::t2 = Eigen::Matrix2d::Zero(2, 2);

Eigen::Matrix3d MatrixEuler::G11 = Eigen::Matrix3d::Zero(3, 3);
Eigen::Matrix2d MatrixEuler::G22 = Eigen::Matrix2d::Zero(2, 2);
Eigen::Matrix<double, 3, 2> MatrixEuler::G12 = Eigen::Matrix<double, 3, 2>::Zero(3, 2);

Eigen::Matrix3d MatrixEuler::G11_dr = Eigen::Matrix3d::Zero(3, 3);
Eigen::Matrix2d MatrixEuler::G22_dr = Eigen::Matrix2d::Zero(2, 2);
Eigen::Matrix<double, 3, 2> MatrixEuler::G12_dr = Eigen::Matrix<double, 3, 2>::Zero(3, 2);

Eigen::Matrix3d MatrixEuler::G11_dT = Eigen::Matrix3d::Zero(3, 3);
Eigen::Matrix2d MatrixEuler::G22_dT = Eigen::Matrix2d::Zero(2, 2);
Eigen::Matrix<double, 3, 2> MatrixEuler::G12_dT = Eigen::Matrix<double, 3, 2>::Zero(3, 2);

Eigen::Vector3d MatrixEuler::pe_vector = Eigen::Vector3d::Zero(3, 1);
Eigen::Vector2d MatrixEuler::p_vector = Eigen::Vector2d::Zero(2, 1);

Eigen::Vector2d MatrixEuler::dH_dp = Eigen::Vector2d::Zero(2, 1);
Eigen::Vector3d MatrixEuler::dH_dpe = Eigen::Vector3d::Zero(3, 1);

Eigen::Matrix3d MatrixEuler::S_matrix = Eigen::Matrix3d::Zero(3, 3);
Eigen::Vector3d MatrixEuler::molecular_dipole = Eigen::Vector3d::Zero(3, 1);

void MatrixEuler::transform_dipole( Eigen::Vector3d & transformed, const double R,
                                                      const double Theta,
                                                      const double phi,
                                                      const double theta,
                                                      const double psi )
{
    double sin_phi = std::sin( phi );
    double cos_phi = std::cos( phi );

    double sin_theta = std::sin( theta );
    double cos_theta = std::cos( theta );

    double sin_psi = std::sin( psi );
    double cos_psi = std::cos( psi );

    S_matrix(0, 0) = cos_psi * cos_phi - cos_theta * sin_phi * sin_psi;
    S_matrix(0, 1) = - sin_psi * cos_phi - cos_theta * sin_phi * cos_psi;
    S_matrix(0, 2) = sin_theta * sin_phi;

    S_matrix(1, 0) = cos_psi * sin_phi + cos_theta * cos_phi * sin_psi;
    S_matrix(1, 1) = - sin_psi * sin_phi + cos_theta * cos_phi * cos_psi;
    S_matrix(1, 2) = - sin_theta * cos_phi;

    S_matrix(2, 0) = sin_theta * sin_psi;
    S_matrix(2, 1) = sin_theta * cos_psi;
    S_matrix(2, 2) = cos_theta;

	// vector of dipole in molecular frame
    molecular_dipole(0) = 0; //AI_IDS.dipx( R, Theta );
    molecular_dipole(1) = 0;
    molecular_dipole(2) = 0; //AI_IDS.dipz( R, Theta );

	// vector of dipole in laboratory frame
    transformed = S_matrix * molecular_dipole;
}

void MatrixEuler::fillIT( const double R2, const double sinT, const double cosT )
// fills Inertia tensor matrix
// input:
// 		R2: R squared
// 		sin_t: sin(Theta)
// 		cos_t: cos(Theta)
{
    inertia_tensor(0, 0) = mu1 * l2 * cosT * cosT + mu2 * R2;
    inertia_tensor(2, 0) = -mu1 * l2 * sinT * cosT;

    inertia_tensor(0, 2) = inertia_tensor(2, 0);
    inertia_tensor(2, 2) = mu1 * l2 * sinT * sinT;

    inertia_tensor(1, 1) = inertia_tensor(0, 0) + inertia_tensor(2, 2);
}

void MatrixEuler::fillITR( const double R )
// fills derivative of Inertia tensor matrix by R
// input:
// 		R variable
{
    inertia_tensor_dr(0, 0) = inertia_tensor_dr(1, 1) = 2 * mu2 * R;
}

void MatrixEuler::fillITT( const double sinT, const double cosT )
// fills derivative of Inertia tensor matrix by Theta
// input:
//  	sin_t: sin(Theta)
// 		cos_t: cos(Theta)
{

    inertia_tensor_dT(0, 0) = - 2 * mu1 * l2 * sinT * cosT;
    inertia_tensor_dT(2, 0) = inertia_tensor_dT(0, 2) = - mu1 * l2 * (cosT * cosT - sinT * sinT);
    inertia_tensor_dT(2, 2) = -inertia_tensor_dT(0, 0);
}

void MatrixEuler::fill_a()
// fills  "a" matrix and it's inverse
// Does not depend on any internal variables, hence called only once in "init" method.
{
    a(0, 0) = mu2;
    a(1, 1) = mu1 * l * l;

    a_inverse = a.inverse();
}

void MatrixEuler::fill_A( )
// fills "A" matrix and it's transpose
// Does not depend on any internal variables, hence called only once in "init" method.
{
    A(1, 1) = mu1 * l * l;

    At = A.transpose();
}

void MatrixEuler::fillW( const double sin_psi, const double cos_psi, const double sin_t, const double cos_t )
// Fills W matrix which will be used in transform of Euler momenta "pe_vector" to vector of angular momenta J:
// 	J = (V^{-1})^T * pe_vector = W * pe_vector
// input:
// 		sin_psi: sin(psi)
// 		cos_psi: cos(psi)
// 		sin_t: sin(theta)
// 		cos_t: cos(theta)
{
    double ctg_t = cos_t / sin_t;

    W(0, 0) = sin_psi / sin_t;
    W(0, 1) = cos_psi;
    W(0, 2) = - sin_psi * ctg_t;

    W(1, 0) = cos_psi / sin_t;
    W(1, 1) = - sin_psi;
    W(1, 2) = - cos_psi * ctg_t;

    W(2, 2) = 1;
}

void MatrixEuler::fillW_dt( const double sin_t, const double cos_t, const double sin_psi, const double cos_psi )
// Fills derivative of W matrix by Euler angle theta
{
    W_dt(0, 0) = - sin_psi * cos_t / sin_t / sin_t;
    W_dt(0, 2) = sin_psi / sin_t / sin_t;

    W_dt(1, 0) = - cos_psi * cos_t / sin_t / sin_t;
    W_dt(1, 2) = cos_psi / sin_t / sin_t;
}

void MatrixEuler::fillW_dpsi( const double sin_psi, const double cos_psi, const double sin_t, const double cos_t )
// Fills derivative of W matrix by Euler angle psi
{
    double ctg_t = cos_t / sin_t;

    W_dpsi(0, 0) = cos_psi / sin_t;
    W_dpsi(0, 1) = - sin_psi;
    W_dpsi(0, 2) = - cos_psi * ctg_t;

    W_dpsi(1, 0) = - sin_psi / sin_t;
    W_dpsi(1, 1) = - cos_psi;
    W_dpsi(1, 2) = sin_psi * ctg_t;
}

double MatrixEuler::hamiltonian( N_Vector y )
{
	double R = NV_Ith_S(y, 0);
	double R2 = R * R;
    
	double Theta = NV_Ith_S(y, 2); 
    double sinT = std::sin(Theta);
    double cosT = std::cos(Theta);

    double theta = NV_Ith_S(y, 6);
    double sin_t = std::sin(theta);
    double cos_t = std::cos(theta);

    double psi = NV_Ith_S(y, 8);
    double sin_psi = std::sin(psi);
    double cos_psi = std::cos(psi);
    
	// filling vector of Euler momenta
    pe_vector(0) = NV_Ith_S(y, 5); 
    pe_vector(1) = NV_Ith_S(y, 7); 
    pe_vector(2) = NV_Ith_S(y, 9); 

    // filling vector of general momenta
    p_vector(0) = NV_Ith_S(y, 1); 
    p_vector(1) = NV_Ith_S(y, 3); 

    fillIT( R2, sinT, cosT );
    fillITR( R );
    fillITT( sinT, cosT );

    // pay attention: passed Euler variables (!)
    fillW( sin_psi, cos_psi, sin_t, cos_t );

    // filling matrices G11, G22, G12
    inertia_tensor_inverse = inertia_tensor.inverse();
    
	G11 = (inertia_tensor - A * a_inverse * At).inverse();
    G22 = (a - At * inertia_tensor_inverse * A).inverse();
    G12 = - G11 * A * a_inverse;
	
	Eigen::Vector3d J_vector = W * pe_vector;

	temp = 0.0;
	temp += 0.5 * J_vector.transpose() * G11 * J_vector;
	temp += 0.5 * p_vector.transpose() * G22 * p_vector;
	temp += J_vector.transpose() * G12 * p_vector;

	return temp;// + AI_PES.pes( R, theta );
}

int MatrixEuler::syst( realtype t, N_Vector y, N_Vector ydot, void * user_data )
// input:
//     R, pR, Theta, pTheta, phi, p_phi, theta, p_theta, psi, p_psi
// output:
//  derivatives \dot{R}, \dot{pR}, \dot{Theta}, \dot{pTheta}, \dot{phi}, \dot{p_phi},
//				\dot{theta}, \dot{p_theta}, \dot{psi}, \dot{p_psi}
{
	// block unused variable warnings
	(void)(t);
	(void)(user_data);

    double R = NV_Ith_S(y, 0);
    double R2 = R * R;

    double Theta = NV_Ith_S(y, 2); 
    double sinT = std::sin(Theta);
    double cosT = std::cos(Theta);

    double theta = NV_Ith_S(y, 6);
    double sin_t = std::sin(theta);
    double cos_t = std::cos(theta);

    double psi = NV_Ith_S(y, 8);
    double sin_psi = std::sin(psi);
    double cos_psi = std::cos(psi);

    // filling vector of Euler momenta
    pe_vector(0) = NV_Ith_S(y, 5); 
    pe_vector(1) = NV_Ith_S(y, 7); 
    pe_vector(2) = NV_Ith_S(y, 9); 

    // filling vector of general momenta
    p_vector(0) = NV_Ith_S(y, 1); 
    p_vector(1) = NV_Ith_S(y, 3); 

    fillIT( R2, sinT, cosT );
    fillITR( R );
    fillITT( sinT, cosT );

    // pay attention: passed Euler variables (!)
    fillW( sin_psi, cos_psi, sin_t, cos_t );

    // filling matrices G11, G22, G12
    inertia_tensor_inverse = inertia_tensor.inverse();
    //t1 = inertia_tensor - A * a_inverse * At;
    G11 = (inertia_tensor - A * a_inverse * At).inverse();

    //t2 = a - At * inertia_tensor_inverse * A;
    G22 = (a - At * inertia_tensor_inverse * A).inverse();

    G12 = - G11 * A * a_inverse;
	
    dH_dp = G22 * p_vector + G12.transpose() * W * pe_vector;

    G11_dr = - G11 * inertia_tensor_dr * G11;
    G12_dr = - G11_dr * A * a_inverse;
    G22_dr = - G22 * At * inertia_tensor_inverse * inertia_tensor_dr * inertia_tensor_inverse * A * G22;

    temp = 0.0;
    temp += 0.5 * pe_vector.transpose() * W.transpose() * G11_dr * W * pe_vector;
    temp += 0.5 * p_vector.transpose() * G22_dr * p_vector;
    temp += pe_vector.transpose() * W.transpose() * G12_dr * p_vector;
    NV_Ith_S(ydot, 1) = - temp;// - AI_PES.dpes_dR(R, Theta); // d(pR)/dt

    G11_dT = - G11 * inertia_tensor_dT * G11;
    G22_dT = - G22 * A.transpose() * inertia_tensor_inverse * inertia_tensor_dT * inertia_tensor_inverse * A * G22;
    G12_dT = -G11_dT * A * a_inverse;

    temp = 0.0;
    temp += 0.5 * pe_vector.transpose() * W.transpose() * G11_dT * W * pe_vector;
    temp += 0.5 * p_vector.transpose() * G22_dT * p_vector;
    temp += pe_vector.transpose() * W.transpose() * G12_dT * p_vector;
    NV_Ith_S(ydot, 3) = - temp;// - AI_PES.dpes_dTheta(R, Theta); // d(pTheta)/dt

    // pay attention: passed Euler variables (!)
    fillW_dt( sin_t, cos_t, sin_psi, cos_psi );

    temp = 0.0;
    temp += 0.5 * pe_vector.transpose() * W_dt.transpose() * G11 * W * pe_vector;
    temp += 0.5 * pe_vector.transpose() * W.transpose() * G11 * W_dt * pe_vector;
    temp += pe_vector.transpose() * W_dt.transpose() * G12 * p_vector;
    NV_Ith_S(ydot, 7) = - temp; // d(pTheta)/dt

    // pay attention: passed Euler variables (!)
    fillW_dpsi( sin_psi, cos_psi, sin_t, cos_t );

    temp = 0.0;
    temp += 0.5 * pe_vector.transpose() * W_dpsi.transpose() * G11 * W * pe_vector;
    temp += 0.5 * pe_vector.transpose() * W.transpose() * G11 * W_dpsi * pe_vector;
    temp += pe_vector.transpose() * W_dpsi.transpose() * G12 * p_vector;
    NV_Ith_S(ydot, 9) = - temp; // d(pPsi)/dt

    dH_dpe = W.transpose() * G11 * W * pe_vector + W.transpose() * G12 * p_vector;

	NV_Ith_S(ydot, 0) = dH_dp(0); // dR/dt 
	NV_Ith_S(ydot, 2) = dH_dp(1); // dTheta/dt 
	NV_Ith_S(ydot, 4) = dH_dpe(0);  // d(phi)/dt 
	NV_Ith_S(ydot, 5) = 0.0; // d(pPhi)/dt
	NV_Ith_S(ydot, 6) = dH_dpe(1); // d(theta)/dt
	NV_Ith_S(ydot, 8) = dH_dpe(2); // d(psi)/dt

	return 0;
}

void MatrixEuler::init()
// initializer method
{
    // filling constant matrices
    fill_a();
    fill_A();
}
