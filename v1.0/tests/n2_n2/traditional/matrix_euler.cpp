#include "matrix_euler.hpp"

//AI_PES_n2_n2 * MatrixEuler::pes = nullptr;
//AI_IDS_n2_n2 * MatrixEuler::ids = nullptr;

// static allocation of matrices
Eigen::Vector3d MatrixEuler::pe_vector = Eigen::Vector3d::Zero(3, 1);
Eigen::Vector4d MatrixEuler::p_vector = Eigen::Vector4d::Zero(4, 1);

Eigen::Matrix3d MatrixEuler::inertia_tensor = Eigen::Matrix3d::Zero(3, 3);
Eigen::Matrix3d MatrixEuler::inertia_tensor_inverse = Eigen::Matrix3d::Zero(3, 3);
Eigen::Matrix3d MatrixEuler::inertia_tensor_dq1 = Eigen::Matrix3d::Zero(3, 3);
Eigen::Matrix3d MatrixEuler::inertia_tensor_dq2 = Eigen::Matrix3d::Zero(3, 3);
Eigen::Matrix3d MatrixEuler::inertia_tensor_dq3 = Eigen::Matrix3d::Zero(3, 3);
Eigen::Matrix3d MatrixEuler::inertia_tensor_dq4 = Eigen::Matrix3d::Zero(3, 3);

Eigen::Matrix4d MatrixEuler::a = Eigen::Matrix4d::Zero(4, 4);
Eigen::Matrix4d MatrixEuler::a_q3 = Eigen::Matrix4d::Zero(4, 4);
Eigen::Matrix4d MatrixEuler::a_inverse = Eigen::Matrix4d::Zero(4, 4);
Eigen::Matrix<double, 3, 4> MatrixEuler::A = Eigen::Matrix<double, 3, 4>::Zero(3, 4);
Eigen::Matrix<double, 4, 3> MatrixEuler::At = Eigen::Matrix<double, 4, 3>::Zero(4, 3);
Eigen::Matrix<double, 3, 4> MatrixEuler::A_q2 = Eigen::Matrix<double, 3, 4>::Zero(3, 4);
Eigen::Matrix<double, 3, 4> MatrixEuler::A_q3 = Eigen::Matrix<double, 3, 4>::Zero(3, 4);

Eigen::Matrix3d MatrixEuler::V = Eigen::Matrix3d::Zero(3, 3);
Eigen::Matrix3d MatrixEuler::W = Eigen::Matrix3d::Zero(3, 3);
Eigen::Matrix3d MatrixEuler::W_dt = Eigen::Matrix3d::Zero(3, 3);
Eigen::Matrix3d MatrixEuler::W_dpsi = Eigen::Matrix3d::Zero(3, 3);

Eigen::Matrix<double, 3, 4> MatrixEuler::t1 = Eigen::Matrix<double, 3, 4>::Zero(3, 4);
Eigen::Matrix<double, 4, 3> MatrixEuler::t2 = Eigen::Matrix<double, 4, 3>::Zero(4, 3);

Eigen::Matrix3d MatrixEuler::G11 = Eigen::Matrix3d::Zero(3, 3);
Eigen::Matrix4d MatrixEuler::G22 = Eigen::Matrix4d::Zero(4, 4);
Eigen::Matrix<double, 3, 4> MatrixEuler::G12 = Eigen::Matrix<double, 3, 4>::Zero(3, 4);

Eigen::Matrix3d MatrixEuler::G11_dq = Eigen::Matrix3d::Zero(3, 3);
Eigen::Matrix4d MatrixEuler::G22_dq = Eigen::Matrix4d::Zero(4, 4);
Eigen::Matrix<double, 3, 4> MatrixEuler::G12_dq = Eigen::Matrix<double, 3, 4>::Zero(3, 4);

Eigen::Vector3d MatrixEuler::J_vector = Eigen::Vector3d::Zero(3, 1);
Eigen::Matrix<double, 1, 3> MatrixEuler::J_vector_t = Eigen::Matrix<double, 1, 3>::Zero(1, 3);
Eigen::Vector4d MatrixEuler::dH_dp = Eigen::Vector4d::Zero(4, 1);
Eigen::Vector3d MatrixEuler::dH_dpe = Eigen::Vector3d::Zero(3, 1);
Eigen::Vector3d MatrixEuler::dH_dJ = Eigen::Vector3d::Zero(3, 1);

double MatrixEuler::temp = 0.0;
const double MatrixEuler::MU1 = masses::N14_AMU / 2.0; // reduced mass of N2
const double MatrixEuler::MU2 = masses::N14_AMU / 2.0; // reduced mass of N2
const double MatrixEuler::MU3 = masses::N2_AMU * masses::N2_AMU / (masses::N2_AMU + masses::N2_AMU); // reduced mass of whole N2-N2 complex

const double MatrixEuler::MU1L1SQ = MatrixEuler::MU1 * lengths::L_N2_ALU * lengths::L_N2_ALU; // mu1 * l1**2
const double MatrixEuler::MU2L2SQ = MatrixEuler::MU2 * lengths::L_N2_ALU * lengths::L_N2_ALU; // mu2 * l2**2

Eigen::Matrix3d MatrixEuler::matrixS = Eigen::Matrix3d::Zero(3, 3);
Eigen::Vector3d MatrixEuler::molecular_dipole = Eigen::Vector3d::Zero(3, 1);

void MatrixEuler::fillS( N_Vector y )
{
    double phi_euler = NV_Ith_S(y, 8);
    double theta_euler = NV_Ith_S(y, 10);
    double psi_euler = NV_Ith_S(y, 12);

    double sin_phi = std::sin( phi_euler );
    double cos_phi = std::cos( phi_euler );

    double sin_theta = std::sin( theta_euler );
    double cos_theta = std::cos( theta_euler );

    double sin_psi = std::sin( psi_euler );
    double cos_psi = std::cos( psi_euler );

    matrixS(0, 0) = cos_psi * cos_phi - cos_theta * sin_phi * sin_psi;
    matrixS(0, 1) = - sin_psi * cos_phi - cos_theta * sin_phi * cos_psi;
    matrixS(0, 2) = sin_theta * sin_phi;

    matrixS(1, 0) = cos_psi * sin_phi + cos_theta * cos_phi * sin_psi;
    matrixS(1, 1) = - sin_psi * sin_phi + cos_theta * cos_phi * cos_psi;
    matrixS(1, 2) = - sin_theta * cos_phi;

    matrixS(2, 0) = sin_theta * sin_psi;
    matrixS(2, 1) = sin_theta * cos_psi;
    matrixS(2, 2) = cos_theta;
}

void MatrixEuler::transform_dipole(Eigen::Vector3d &dip, N_Vector y)
{
    fillS( y );

    double R = NV_Ith_S(y, 0);
    double Theta1 = NV_Ith_S(y, 2);
    double Phi = NV_Ith_S(y, 4);
    double Theta2 = NV_Ith_S(y, 6);

    //molecular_dipole(0) = ids->dipx( R, Theta1, Theta2, Phi );
    //molecular_dipole(1) = ids->dipy( R, Theta1, Theta2, Phi );
    //molecular_dipole(2) = ids->dipz( R, Theta1, Theta2, Phi );

    dip = matrixS * molecular_dipole;
}

void MatrixEuler::hamilton_to_lagrange_variables(N_Vector y, N_Vector out)
{
    AngleData q1( NV_Ith_S(y, 2) );
    AngleData q2( NV_Ith_S(y, 4) );
    AngleData q3( NV_Ith_S(y, 6) );
    double q4 = NV_Ith_S(y, 0);
    double q4_sq = q4 * q4;

    AngleData phi_euler( NV_Ith_S(y, 8) );
    AngleData theta_euler( NV_Ith_S(y, 10) );
    AngleData psi_euler( NV_Ith_S(y, 12));

    // filling vector of general momenta
    p_vector(0) = NV_Ith_S(y, 3); // pTheta1
    p_vector(1) = NV_Ith_S(y, 5); // pPhi
    p_vector(2) = NV_Ith_S(y, 7); // pTheta2
    p_vector(3) = NV_Ith_S(y, 1); // pR

    // filling vector of Euler momenta
    pe_vector(0) = NV_Ith_S(y, 9); // pPhi_euler
    pe_vector(1) = NV_Ith_S(y, 11); // pTheta_euler
    pe_vector(2) = NV_Ith_S(y, 13); // pPsi_euler

    fillIT( &q1, &q2, &q3, q4_sq );
    fill_a( &q3 );
    fill_A( &q2, &q3 );
    fillW( &theta_euler, &psi_euler );

    G11 = (inertia_tensor - A * a_inverse * At).inverse();
    G22 = (a - At * inertia_tensor_inverse * A).inverse();
    G12 = - G11 * A * a_inverse;

    J_vector = W * pe_vector;

    dH_dp = G22 * p_vector + G12.transpose() * J_vector; //qdot
    dH_dpe = W.transpose() * G11 * J_vector + W.transpose() * G12 * p_vector; // eulerdot

    NV_Ith_S(out, 0) = NV_Ith_S(y, 0); // R
    NV_Ith_S(out, 1) = dH_dp(3); // Rdot
    NV_Ith_S(out, 2) = NV_Ith_S(y, 2); //Theta1
    NV_Ith_S(out, 3) = dH_dp(0); // Theta1dot
    NV_Ith_S(out, 4) = NV_Ith_S(y, 4); // Phi
    NV_Ith_S(out, 5) = dH_dp(1); // Phidot
    NV_Ith_S(out, 6) = NV_Ith_S(y, 6); // Theta2
    NV_Ith_S(out, 7) = dH_dp(2); // Theta2dot
    NV_Ith_S(out, 8) = NV_Ith_S(y, 8);
    NV_Ith_S(out, 9) = dH_dpe(0);
    NV_Ith_S(out, 10) = NV_Ith_S(y, 10);
    NV_Ith_S(out, 11) = dH_dpe(1);
    NV_Ith_S(out, 12) = NV_Ith_S(y, 12);
    NV_Ith_S(out, 13) = dH_dpe(2);
}


void MatrixEuler::exchange_angles_lagrange(N_Vector y, N_Vector out)
{
    NV_Ith_S(out, 0) = NV_Ith_S(y, 0); // R
    NV_Ith_S(out, 1) = NV_Ith_S(y, 1); // Rdot
    NV_Ith_S(out, 2) = M_PI - NV_Ith_S(y, 6); // Theta1 -> Pi - Theta2
    NV_Ith_S(out, 3) = -NV_Ith_S(y, 7); // Theta1dot -> -Theta2dot
    NV_Ith_S(out, 4) = NV_Ith_S(y, 4); // Phi
    NV_Ith_S(out, 5) = NV_Ith_S(y, 5); // Phidot
    NV_Ith_S(out, 6) = M_PI - NV_Ith_S(y, 2); // Theta2 -> Pi - Theta1
    NV_Ith_S(out, 7) = -NV_Ith_S(y, 3); // Theta2dot -> -Theta1dot
    NV_Ith_S(out, 8) = M_PI + NV_Ith_S(y, 8); // PhiEuler -> Pi + PhiEuler
    NV_Ith_S(out, 9) = NV_Ith_S(y, 9); // PhiEuler_dot
    NV_Ith_S(out, 10) = M_PI - NV_Ith_S(y, 10); // ThetaEuler -> Pi - ThetaEuler
    NV_Ith_S(out, 11) = -NV_Ith_S(y, 11); // ThetaEuler_dot -> -ThetaEuler_dot
    NV_Ith_S(out, 12) = M_PI - NV_Ith_S(y, 12) - NV_Ith_S(y, 4); // PsiEuler -> Pi - PsiEuler - Phi
    NV_Ith_S(out, 13) = -NV_Ith_S(y, 13) - NV_Ith_S(y, 5); // PsiEuler_dot -> -PsiEuler_dot - Phidot
}

void MatrixEuler::lagrange_to_hamilton_variables(N_Vector y, N_Vector out)
{
    AngleData q1( NV_Ith_S(y, 2) );
    AngleData q2( NV_Ith_S(y, 4) );
    AngleData q3( NV_Ith_S(y, 6) );
    double q4 = NV_Ith_S(y, 0);
    double q4_sq = q4 * q4;

    AngleData phi_euler( NV_Ith_S(y, 8) );
    AngleData theta_euler( NV_Ith_S(y, 10) );
    AngleData psi_euler( NV_Ith_S(y, 12));

    Eigen::Vector4d qdot;
    qdot(0) = NV_Ith_S(y, 3); // Theta1dot
    qdot(1) = NV_Ith_S(y, 5); // Phidot
    qdot(2) = NV_Ith_S(y, 7); // Theta2dot
    qdot(3) = NV_Ith_S(y, 1); // Rdot

    Eigen::Vector3d eulerdot;
    eulerdot(0) = NV_Ith_S(y, 9); // PhiEuler_dot
    eulerdot(1) = NV_Ith_S(y, 11); // ThetaEuler_dot
    eulerdot(2) = NV_Ith_S(y, 13); // PsiEuler_dot

    fillIT( &q1, &q2, &q3, q4_sq );
    fill_a( &q3 );
    fill_A( &q2, &q3 );
    fillV( &theta_euler, &psi_euler );

    p_vector = A.transpose() * V * eulerdot + a * qdot;
    pe_vector = V.transpose() * inertia_tensor * V * eulerdot + V.transpose() * A * qdot;

    NV_Ith_S(out, 0) = NV_Ith_S(y, 0); // R
    NV_Ith_S(out, 1) = p_vector(3); // pR
    NV_Ith_S(out, 2) = NV_Ith_S(y, 2); // Theta1
    NV_Ith_S(out, 3) = p_vector(0); // pTheta1
    NV_Ith_S(out, 4) = NV_Ith_S(y, 4); // Phi
    NV_Ith_S(out, 5) = p_vector(1); // pPhi
    NV_Ith_S(out, 6) = NV_Ith_S(y, 6); // Theta2
    NV_Ith_S(out, 7) = p_vector(2); // pTheta2
    NV_Ith_S(out, 8) = NV_Ith_S(y, 8); // PhiEuler
    NV_Ith_S(out, 9) = pe_vector(0); // pPhiEuler
    NV_Ith_S(out, 10) = NV_Ith_S(y, 10); // ThetaEuler
    NV_Ith_S(out, 11) = pe_vector(1); // pThetaEuler
    NV_Ith_S(out, 12) = NV_Ith_S(y, 12); // PsiEuler
    NV_Ith_S(out, 13) = pe_vector(2); // pPsiEuler
}

void MatrixEuler::init( )
{
    // calculating coordinate-independent elements of "a" matrix
    a(0, 0) = MU1L1SQ;
    a(2, 2) = MU2L2SQ;
    a(3, 3) = MU3;

    A(1, 0) = MU1L1SQ;
}

void MatrixEuler::fillIT( const AngleData * q1, const AngleData * q2, const AngleData * q3, const double q4_sq )
// Fills inertia tensor matrix and its' inverse
{
    inertia_tensor(0, 0) = MU1L1SQ * q1->cos_sq + MU2L2SQ * (q2->sin_sq * q3->sin_sq + q3->cos_sq) + MU3 * q4_sq;
    inertia_tensor(1, 0) = -MU2L2SQ * q2->sin * q2->cos * q3->sin_sq;
    inertia_tensor(2, 0) = -MU1L1SQ * q1->sin * q1->cos - MU2L2SQ * q2->cos * q3->sin * q3->cos;

    inertia_tensor(0, 1) = inertia_tensor(1, 0);
    inertia_tensor(1, 1) = MU1L1SQ + MU2L2SQ * (q2->cos_sq * q3->sin_sq + q3->cos_sq) + MU3 * q4_sq;
    inertia_tensor(2, 1) = -MU2L2SQ * q2->sin * q3->sin * q3->cos;

    inertia_tensor(0, 2) = inertia_tensor(2, 0);
    inertia_tensor(1, 2) = inertia_tensor(2, 1);
    inertia_tensor(2, 2) = MU1L1SQ * q1->sin_sq + MU2L2SQ * q3->sin_sq;

    inertia_tensor_inverse = inertia_tensor.inverse();
}

void MatrixEuler::fillIT_dq1( const AngleData * q1 )
// Fills derivative of inertia tensor matrix by q1 variable
{
    inertia_tensor_dq1(0, 0) = -2.0 * MU1L1SQ * q1->sin * q1->cos;
    inertia_tensor_dq1(0, 2) = - MU1L1SQ * (q1->cos_sq  - q1->sin_sq);
    inertia_tensor_dq1(2, 0) = inertia_tensor_dq1(0, 2);
    inertia_tensor_dq1(2, 2) = - inertia_tensor_dq1(0, 0);
}

void MatrixEuler::fillIT_dq2( const AngleData * q2, const AngleData * q3 )
// Fills derivative of inertia tensor matrix by q2 variable
{
    inertia_tensor_dq2(0, 0) = 2.0 * MU2L2SQ * q2->sin * q2->cos * q3->sin_sq;
    inertia_tensor_dq2(0, 1) = MU2L2SQ * q3->sin_sq * (q2->sin_sq - q2->cos_sq);
    inertia_tensor_dq2(0, 2) = MU2L2SQ * q2->sin * q3->sin * q3->cos;

    inertia_tensor_dq2(1, 0) = inertia_tensor_dq2(0, 1);
    inertia_tensor_dq2(1, 1) = -2.0 * MU2L2SQ * q2->sin * q2->cos * q3->sin_sq;
    inertia_tensor_dq2(1, 2) = - MU2L2SQ * q2->cos * q3->sin * q3->cos;

    inertia_tensor_dq2(2, 0) = inertia_tensor_dq2(0, 2);
    inertia_tensor_dq2(2, 1) = inertia_tensor_dq2(1, 2);
}

void MatrixEuler::fillIT_dq3( const AngleData * q2, const AngleData * q3 )
// Fills derivative of inertia tensor matrix by q3 variable
{
    inertia_tensor_dq3(0, 0) = -2.0 * MU2L2SQ * q2->cos_sq * q3->sin * q3->cos;
    inertia_tensor_dq3(0, 1) = -2.0 * MU2L2SQ * q2->sin * q2->cos * q3->sin * q3->cos;
    inertia_tensor_dq3(0, 2) = - MU2L2SQ * q2->cos * (q3->cos_sq - q3->sin_sq);

    inertia_tensor_dq3(1, 0) = inertia_tensor_dq3(0, 1);
    inertia_tensor_dq3(1, 1) = -2.0 * MU2L2SQ * q3->sin * q3->cos * q2->sin_sq;
    inertia_tensor_dq3(1, 2) = -MU2L2SQ * q2->sin * (q3->cos_sq - q3->sin_sq);

    inertia_tensor_dq3(2, 0) = inertia_tensor_dq3(0, 2);
    inertia_tensor_dq3(2, 1) = inertia_tensor_dq3(1, 2);
    inertia_tensor_dq3(2, 2) = 2.0 * MU2L2SQ * q3->sin * q3->cos;

}

void MatrixEuler::fillIT_dq4( const double q4 )
// Fills derivative of inertia tensor matrix by q4 variable
{
   inertia_tensor_dq4(0, 0) = 2.0 * MU3 * q4;
   inertia_tensor_dq4(1, 1) = inertia_tensor_dq4(0, 0);
}

void MatrixEuler::fill_a( const AngleData * q3 )
// Recalculates the only elements of "a" matrix that depend on internal coordinates
// Calculates "a" inverse matrix (a_inverse)
{
    a(1, 1) = MU2L2SQ * q3->sin_sq;

    a_inverse = a.inverse();
}

void MatrixEuler::fill_a_q3( const AngleData * q3 )
// Recalculates the only elements of derivative of "a" matrix by q3 variable
{
    a_q3(1, 1) = 2.0 * MU2L2SQ * q3->sin * q3->cos;
}

void MatrixEuler::fill_A( const AngleData * q2, const AngleData * q3 )
// Recalculates the only elements of "A" matrix that depend on internal coordinates
// Fills "A" transposed matrix (At)
{
    A(0, 1) = - MU2L2SQ * q2->cos * q3->sin * q3->cos;
    A(1, 1) = - MU2L2SQ * q2->sin * q3->sin * q3->cos;
    A(2, 1) = MU2L2SQ * q3->sin_sq;

    A(0, 2) = - MU2L2SQ * q2->sin;
    A(1, 2) = MU2L2SQ * q2->cos;

    At = A.transpose();
}

void MatrixEuler::fill_A_q2( const AngleData * q2, const AngleData * q3 )
// Fills derivative of "A" matrix by q2 variable
{
    A_q2(0, 1) =  MU2L2SQ * q2->sin * q3->sin * q3->cos;
    A_q2(0, 2) = -MU2L2SQ * q2->cos;
    A_q2(1, 1) = -MU2L2SQ * q2->cos * q3->sin * q3->cos;
    A_q2(1, 2) = -MU2L2SQ * q2->sin;
}

void MatrixEuler::fill_A_q3( const AngleData * q2, const AngleData * q3 )
// Fills derivative of "A" matrix by q3 variable
{
    A_q3(0, 1) = -MU2L2SQ * q2->cos * (q3->cos_sq - q3->sin_sq);
    A_q3(1, 1) = -MU2L2SQ * q2->sin * (q3->cos_sq - q3->sin_sq);
    A_q3(2, 1) = 2 * MU2L2SQ * q3->sin * q3->cos;
}

void MatrixEuler::fillV( const AngleData * theta, const AngleData * psi )
{
    V(0, 0) = theta->sin * psi->sin;
    V(0, 1) = psi->cos;
    V(0, 2) = 0.0;

    V(1, 0) = theta->sin * psi->cos;
    V(1, 1) = -psi->sin;
    V(1, 2) = 0.0;

    V(2, 0) = theta->cos;
    V(2, 1) = 0.0;
    V(2, 2) = 1.0;
}

void MatrixEuler::fillW( const AngleData * theta, const AngleData * psi )
// Fills W matrix which will be used in transform of Euler momenta "pe_vector" to vector of angular momenta J:
// 	J = (V^{-1})^T * pe_vector = W * pe_vector
// input:
// 		AngleData * theta: sin(theta), cos(theta)
// 		AngleData * psi: sin(psi), cos(psi)
{
    double ctg_theta = theta->cos / theta->sin;

    W(0, 0) = psi->sin / theta->sin;
    W(0, 1) = psi->cos;
    W(0, 2) = - psi->sin * ctg_theta;

    W(1, 0) = psi->cos / theta->sin;
    W(1, 1) = - psi->sin;
    W(1, 2) = - psi->cos * ctg_theta;

    W(2, 2) = 1.0;
}

void MatrixEuler::fillW_dt( const AngleData * theta, const AngleData * psi )
// Fills derivative of W matrix by Euler angle theta
// input:
// 		AngleData * theta: sin(theta), cos(theta)
// 		AngleData * psi: sin(psi), cos(psi)
{
    W_dt(0, 0) = - psi->sin * theta->cos / theta->sin_sq;
    W_dt(0, 2) = psi->sin / theta->sin_sq;

    W_dt(1, 0) = - psi->cos * theta->cos / theta->sin_sq;
    W_dt(1, 2) = psi->cos / theta->sin_sq;
}

void MatrixEuler::fillW_dpsi( const AngleData * theta, const AngleData * psi )
// Fills derivative of W matrix by Euler angle psi
// input:
// 		AngleData * theta: sin(theta), cos(theta)
// 		AngleData * psi: sin(psi), cos(psi)
{
    double ctg_theta = theta->cos / theta->sin;

    W_dpsi(0, 0) = psi->cos / theta->sin;
    W_dpsi(0, 1) = - psi->sin;
    W_dpsi(0, 2) = - psi->cos * ctg_theta;

    W_dpsi(1, 0) = - psi->sin / theta->sin;
    W_dpsi(1, 1) = - psi->cos;
    W_dpsi(1, 2) = psi->sin * ctg_theta;
}

Eigen::Vector3d MatrixEuler::angular_momentum( N_Vector y )
{
    AngleData theta_euler( NV_Ith_S(y, 10) );
    AngleData psi_euler( NV_Ith_S(y, 12));
    // using Euler angles to fill W matrix
    fillW( &theta_euler, &psi_euler );

    // filling vector of Euler momenta
    pe_vector(0) = NV_Ith_S(y, 9); // pPhi_euler
    pe_vector(1) = NV_Ith_S(y, 11); // pTheta_euler
    pe_vector(2) = NV_Ith_S(y, 13); // pPsi_euler

    return W * pe_vector; 
}

double MatrixEuler::angular_momentum_length( N_Vector y )
{
    AngleData theta_euler( NV_Ith_S(y, 10) );
    AngleData psi_euler( NV_Ith_S(y, 12));
    // using Euler angles to fill W matrix
    fillW( &theta_euler, &psi_euler );

    // filling vector of Euler momenta
    pe_vector(0) = NV_Ith_S(y, 9); // pPhi_euler
    pe_vector(1) = NV_Ith_S(y, 11); // pTheta_euler
    pe_vector(2) = NV_Ith_S(y, 13); // pPsi_euler

    J_vector = W * pe_vector;

    return std::sqrt(J_vector(0)*J_vector(0) + J_vector(1)*J_vector(1) + J_vector(2)*J_vector(2));
}

int MatrixEuler::syst( realtype t, N_Vector y, N_Vector ydot, void * user_data )
// input: 14 variables
//  R := q4, pR, Theta1(N2) := q1, pTheta1(N2), Phi := q2, pPhi, Theta2(N2) := q3, pTheta2(N2),
//		  phi_euler, pPhi_euler, theta_euler, pTheta_euler, psi_euler, pPsi_euler
// y[0] = q4 := R
// y[2] = q1 := Theta1(N2)
// y[4] = q2 := Phi
// y[6] = q3 := Theta2(N2)matrix
//
// y[8] = phi_euler
// y[10] = theta_euler
// y[12] = psi_euler
//
// output:
//    derivatives of all variables in given order
{
    (void)( t );
    (void)( user_data );

	//std::cout << "R: " << NV_Ith_S(y, 0) << std::endl;
    std::cout << std::fixed << std::setprecision(16);

    AngleData q1( NV_Ith_S(y, 2) );
    AngleData q2( NV_Ith_S(y, 4) );
    AngleData q3( NV_Ith_S(y, 6) );
    double q4 = NV_Ith_S(y, 0);
    double q4_sq = q4 * q4;

    AngleData theta_euler( NV_Ith_S(y, 10) );
    AngleData psi_euler( NV_Ith_S(y, 12));

    // filling vector of general momenta
    p_vector(0) = NV_Ith_S(y, 3); // pTheta1 (CO2)
    p_vector(1) = NV_Ith_S(y, 5); // pPhi
    p_vector(2) = NV_Ith_S(y, 7); // pTheta2 (H2)
    p_vector(3) = NV_Ith_S(y, 1); // pR

    // filling vector of Euler momenta
    pe_vector(0) = NV_Ith_S(y, 9); // pPhi_euler
    pe_vector(1) = NV_Ith_S(y, 11); // pTheta_euler
    pe_vector(2) = NV_Ith_S(y, 13); // pPsi_euler

    fillIT( &q1, &q2, &q3, q4_sq );
    //std::cout << "IT: " << std::endl << inertia_tensor << std::endl;

    fillIT_dq1( &q1 );
    //std::cout << "IT_dq1: " << std::endl << inertia_tensor_dq1 << std::endl;
    
    fillIT_dq2( &q2, &q3 );
    //std::cout << "IT_dq2: " << std::endl << inertia_tensor_dq2 << std::endl;
    
    fillIT_dq3( &q2, &q3 );
    //std::cout << "IT_dq3: " << std::endl << inertia_tensor_dq3 << std::endl;
    
    fillIT_dq4( q4 );
    //std::cout << "IT_dq4: " << std::endl << inertia_tensor_dq4 << std::endl;

    fill_a( &q3 );
    //std::cout << "a: " << std::endl << a << std::endl;

    fill_a_q3( &q3 );
    //std::cout << "a_dq3: " << std::endl << a_q3 << std::endl;

    fill_A( &q2, &q3 );
    //std::cout << "A: " << std::endl << A << std::endl;

    fill_A_q2( &q2, &q3 );
    //std::cout << "A_dq2: " << std::endl << A_q2 << std::endl;
    
    fill_A_q3( &q2, &q3 );
    //std::cout << "A_dq3: " << std::endl << A_q3 << std::endl;

    // using Euler angles to fill W matrix
    fillW( &theta_euler, &psi_euler );

    G11 = (inertia_tensor - A * a_inverse * At).inverse();
    G22 = (a - At * inertia_tensor_inverse * A).inverse();
    G12 = - G11 * A * a_inverse;

    //std::cout << "G11: " << std::endl << G11 << std::endl;
    //std::cout << "G12: " << std::endl << G12 << std::endl;
    //std::cout << "G22: " << std::endl << G22 << std::endl;

    J_vector.noalias() = W * pe_vector;

    J_vector_t = J_vector.transpose();

    dH_dp = G22 * p_vector + G12.transpose() * J_vector;

    NV_Ith_S(ydot, 0) = dH_dp(3); // dR / dt
    NV_Ith_S(ydot, 2) = dH_dp(0); // dTheta1(CO2) / dt
    NV_Ith_S(ydot, 4) = dH_dp(1); // dPhi / dt
    NV_Ith_S(ydot, 6) = dH_dp(2); // dTheta2(H2) / dt

    // G11_dq, G12_dq, G22_dq -- temporary storage for derivatives of corresponding matrices
    // dR = dq[4]
    t1.noalias() = inertia_tensor_inverse * A * G22; // .noalias() - не храним результат промежуточных вычислений
    t2.noalias() = G22 * At * inertia_tensor_inverse;

    G11_dq.noalias() = - G11 * inertia_tensor_dq4 * G11;
    G12_dq.noalias() = - G11_dq * A * a_inverse;
    G22_dq.noalias() = - t2 * inertia_tensor_dq4 * t1;

    temp = 0.0;
    temp += 0.5 * J_vector_t * G11_dq * J_vector;
    temp += 0.5 * p_vector.transpose() * G22_dq * p_vector;
    temp += J_vector_t * G12_dq * p_vector;
    NV_Ith_S(ydot, 1) = - temp; // - pes->dpesdR( NV_Ith_S(y, 0), NV_Ith_S(y, 2), NV_Ith_S(y, 6), NV_Ith_S(y, 4) ); // d(pR)/dt

    // dTheta1(CO2) = dq[1]
    G11_dq.noalias() = - G11 * inertia_tensor_dq1 * G11;
    G12_dq.noalias() = - G11_dq * A * a_inverse;
    G22_dq.noalias() = - t2 * inertia_tensor_dq1 * t1;

    temp = 0.0;
    temp += 0.5 * J_vector_t * G11_dq * J_vector;
    temp += 0.5 * p_vector.transpose() * G22_dq * p_vector;
    temp += J_vector_t * G12_dq * p_vector;
    NV_Ith_S(ydot, 3) = - temp; // - pes->dpesdTheta1( NV_Ith_S(y, 0), NV_Ith_S(y, 2), NV_Ith_S(y, 6), NV_Ith_S(y, 4) ); // d(pTheta1)/dt

    // dPhi = dq2
    G11_dq.noalias() = - G11 * (inertia_tensor_dq2 - \
                                A_q2 * a_inverse * At - A * a_inverse * A_q2.transpose()) * G11;
    G12_dq.noalias() = - G11_dq * A * a_inverse - G11 * A_q2 * a_inverse;
    G22_dq.noalias() = - G22 * (- A_q2.transpose() * inertia_tensor_inverse * A + \
               At * inertia_tensor_inverse * inertia_tensor_dq2 * inertia_tensor_inverse * A  - \
                                At * inertia_tensor_inverse * A_q2) * G22;

    temp = 0.0;
    temp += 0.5 * J_vector_t * G11_dq * J_vector;
    temp += 0.5 * p_vector.transpose() * G22_dq * p_vector;
    temp += J_vector_t * G12_dq * p_vector;
    NV_Ith_S(ydot, 5) = - temp; // - pes->dpesdPhi( NV_Ith_S(y, 0), NV_Ith_S(y, 2), NV_Ith_S(y, 6), NV_Ith_S(y, 4) ); // d(pPhi)/dt

    // dTheta2(H2) = dq3
    G11_dq.noalias() = - G11 * \
        (inertia_tensor_dq3 - A_q3 * a_inverse * At + A * a_inverse * a_q3 * a_inverse * At - \
        A * a_inverse * A_q3.transpose()) * G11;
    G12_dq.noalias() = - G11_dq * A * a_inverse - G11 * A_q3 * a_inverse + G11 * A * a_inverse * a_q3 * a_inverse;
    G22_dq.noalias() = -G22 * (a_q3 - A_q3.transpose() * inertia_tensor_inverse * A + \
        At * inertia_tensor_inverse * inertia_tensor_dq3 * inertia_tensor_inverse * A - \
        At * inertia_tensor_inverse * A_q3 ) * G22;

    temp = 0.0;
    temp += 0.5 * J_vector_t * G11_dq * J_vector;
    temp += 0.5 * p_vector.transpose() * G22_dq * p_vector;
    temp += J_vector_t * G12_dq * p_vector;
    NV_Ith_S(ydot, 7) = - temp; // - pes->dpesdTheta2( NV_Ith_S(y, 0), NV_Ith_S(y, 2), NV_Ith_S(y, 6), NV_Ith_S(y, 4) ); // d(pTheta2)/dt

    // pay attention: passed Euler variables (!)
    fillW_dt( &theta_euler, &psi_euler );

    temp = 0.0;
    temp += 0.5 * pe_vector.transpose() * W_dt.transpose() * G11 * J_vector;
    temp += 0.5 * J_vector_t * G11 * W_dt * pe_vector;
    temp += pe_vector.transpose() * W_dt.transpose() * G12 * p_vector;
    NV_Ith_S(ydot, 11) = - temp;// d(pTheta_euler)/dt

    // pay attention: passed Euler variables (!)
    fillW_dpsi( &theta_euler, &psi_euler );

    temp = 0.0;
    temp += 0.5 * pe_vector.transpose() * W_dpsi.transpose() * G11 * J_vector;
    temp += 0.5 * J_vector_t * G11 * W_dpsi * pe_vector;
    temp += pe_vector.transpose() * W_dpsi.transpose() * G12 * p_vector;
    NV_Ith_S(ydot, 13) = - temp; // d(pPsi_euler)/dt

    NV_Ith_S(ydot, 9) = 0.0; // d(pPhi_euler) / dt

    dH_dpe = W.transpose() * G11 * J_vector + W.transpose() * G12 * p_vector;

    NV_Ith_S(ydot, 8) = dH_dpe(0); // d(phi_euler)/dt
    NV_Ith_S(ydot, 10) = dH_dpe(1); // d(theta_euler)/dt
    NV_Ith_S(ydot, 12) = dH_dpe(2); // d(psi_euler)/dt

    return 0;
}

double MatrixEuler::lagrangian( N_Vector y )
// input:
// R := q4, Rdot, Theta1 := q1, Theta1dot, Phi := q2, Phidot, Theta2 := q3, Theta2dot,
//      phi_euler, phi_euler_dot, theta_euler, theta_euler_dot, psi_euler, psi_euler_dot
{
    AngleData q1( NV_Ith_S(y, 2) );
    AngleData q2( NV_Ith_S(y, 4) );
    AngleData q3( NV_Ith_S(y, 6) );
    double q4 = NV_Ith_S(y, 0);
    double q4_sq = q4 * q4;

    AngleData theta_euler( NV_Ith_S(y, 10) );
    AngleData psi_euler( NV_Ith_S(y, 12));

    Eigen::Vector4d qdot_vector;
    qdot_vector(0) = NV_Ith_S(y, 3); // Theta1dot
    qdot_vector(1) = NV_Ith_S(y, 5); // Phidot
    qdot_vector(2) = NV_Ith_S(y, 7); // Theta2dot
    qdot_vector(3) = NV_Ith_S(y, 1); // Rdot

    Eigen::Vector3d eulerdot_vector;
    eulerdot_vector(0) = NV_Ith_S(y, 9); // phi_euler_dot
    eulerdot_vector(1) = NV_Ith_S(y, 11); // theta_euler_dot
    eulerdot_vector(2) = NV_Ith_S(y, 13); // psi_euler_dot

    fillIT( &q1, &q2, &q3, q4_sq );
    fill_a( &q3 );
    fill_A( &q2, &q3 );

    fillV( &theta_euler, &psi_euler );
    Eigen::Vector3d Omega = V * eulerdot_vector;

    temp = 0.0;
    temp += 0.5 * Omega.transpose() * inertia_tensor * Omega;
    temp += Omega.transpose() * A * qdot_vector;
    temp += 0.5 * qdot_vector.transpose() * a * qdot_vector;

    return temp; // - pes->pes( NV_Ith_S(y, 0), NV_Ith_S(y, 2), NV_Ith_S(y, 6), NV_Ith_S(y, 4) );
}

double MatrixEuler::hamiltonian( N_Vector y )
// input: 14 variables
//  R := q4, pR, Theta1(CO2) := q1, pTheta1(CO2), Phi := q2, pPhi, Theta2(H2) := q3, pTheta2,
//		  phi_euler, pPhi_euler, theta_euler, pTheta_euler, psi_euler, pPsi_euler
// output:
//    Hamiltonian value
{
    AngleData q1( NV_Ith_S(y, 2) );
    AngleData q2( NV_Ith_S(y, 4) );
    AngleData q3( NV_Ith_S(y, 6) );
    double q4 = NV_Ith_S(y, 0);
    double q4_sq = q4 * q4;

    AngleData theta_euler( NV_Ith_S(y, 10) );
    AngleData psi_euler( NV_Ith_S(y, 12));

    // filling vector of general momenta
    p_vector(0) = NV_Ith_S(y, 3); // pTheta1 (CO2)
    p_vector(1) = NV_Ith_S(y, 5); // pPhi
    p_vector(2) = NV_Ith_S(y, 7); // pTheta2 (H2)
    p_vector(3) = NV_Ith_S(y, 1); // pR

    // filling vector of Euler momenta
    pe_vector(0) = NV_Ith_S(y, 9); // pPhi_euler
    pe_vector(1) = NV_Ith_S(y, 11); // pTheta_euler
    pe_vector(2) = NV_Ith_S(y, 13); // pPsi_euler

    fillIT( &q1, &q2, &q3, q4_sq );
    fill_a( &q3 );
    fill_A( &q2, &q3 );

    G11 = (inertia_tensor - A * a_inverse * At).inverse();
    G22 = (a - At * inertia_tensor_inverse * A).inverse();
    G12 = -G11 * A * a_inverse;

    fillW( &theta_euler, &psi_euler );
    J_vector = W * pe_vector;

    temp = 0.0;
    temp += 0.5 * J_vector.transpose() * G11 * J_vector;
    temp += 0.5 * p_vector.transpose() * G22 * p_vector;
    temp += J_vector.transpose() * G12 * p_vector;

    double *ydata = N_VGetArrayPointer( y );
    return temp; //+ pes->pes( ydata[0], ydata[2], ydata[6], ydata[4] );
}
