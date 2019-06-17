#pragma once

#include <vector>

#include <Eigen/Dense>

#include "constants.hpp"
#include "masses.hpp"
#include "QData.hpp"

const double MU1 = masses::N14_AMU / 2.0; // reduced mass of N2
const double MU2 = masses::N14_AMU / 2.0; // reduced mass of N2
const double MU3 = masses::N2_AMU * masses::N2_AMU / (masses::N2_AMU + masses::N2_AMU); // reduced mass of whole N2-N2 complex

const double MU1L1SQ = MU1 * lengths::L_N2_ALU * lengths::L_N2_ALU; // mu1 * l1**2
const double MU2L2SQ = MU2 * lengths::L_N2_ALU * lengths::L_N2_ALU; // mu2 * l2**2

void fill_IT( Eigen::Ref<Eigen::Matrix<double, IT_SIZE, IT_SIZE>> m, std::vector<QData> const& q )
{
    m(0, 0) = MU1L1SQ * q[0].cos_sq + MU2L2SQ * (q[1].sin_sq * q[2].sin_sq + q[2].cos_sq) + MU3 * q[3].sq;
    m(1, 0) = -MU2L2SQ * q[1].sin * q[1].cos * q[2].sin_sq;
    m(2, 0) = -MU1L1SQ * q[0].sin * q[0].cos - MU2L2SQ * q[1].cos * q[2].sin * q[2].cos;

    m(0, 1) = m(1, 0);
    m(1, 1) = MU1L1SQ + MU2L2SQ * (q[1].cos_sq * q[2].sin_sq + q[2].cos_sq) + MU3 * q[3].sq;
    m(2, 1) = -MU2L2SQ * q[1].sin * q[2].sin * q[2].cos;

    m(0, 2) = m(2, 0);
    m(1, 2) = m(2, 1);
    m(2, 2) = MU1L1SQ * q[0].sin_sq + MU2L2SQ * q[2].sin_sq;
}

void fill_IT_dq1( Eigen::Ref<Eigen::Matrix<double, IT_SIZE, IT_SIZE>> m, std::vector<QData> const& q )
{
    m(0, 0) = -2.0 * MU1L1SQ * q[0].sin * q[0].cos;
    m(0, 2) = - MU1L1SQ * (q[0].cos_sq  - q[0].sin_sq);
    m(2, 0) = m(0, 2);
    m(2, 2) = - m(0, 0);
}

void fill_IT_dq2( Eigen::Ref<Eigen::Matrix<double, IT_SIZE, IT_SIZE>> m, std::vector<QData> const& q )
{
    m(0, 0) = 2.0 * MU2L2SQ * q[1].sin * q[1].cos * q[2].sin_sq;
    m(0, 1) = MU2L2SQ * q[2].sin_sq * (q[1].sin_sq - q[1].cos_sq);
    m(0, 2) = MU2L2SQ * q[1].sin * q[2].sin * q[2].cos;

    m(1, 0) = m(0, 1);
    m(1, 1) = -2.0 * MU2L2SQ * q[1].sin * q[1].cos * q[2].sin_sq;
    m(1, 2) = - MU2L2SQ * q[1].cos * q[2].sin * q[2].cos;

    m(2, 0) = m(0, 2);
    m(2, 1) = m(1, 2);
}

void fill_IT_dq3( Eigen::Ref<Eigen::Matrix<double, IT_SIZE, IT_SIZE>> m, std::vector<QData> const& q )
{
    m(0, 0) = -2.0 * MU2L2SQ * q[1].cos_sq * q[2].sin * q[2].cos;
    m(0, 1) = -2.0 * MU2L2SQ * q[1].sin * q[1].cos * q[2].sin * q[2].cos;
    m(0, 2) = - MU2L2SQ * q[1].cos * (q[2].cos_sq - q[2].sin_sq);

    m(1, 0) = m(0, 1);
    m(1, 1) = -2.0 * MU2L2SQ * q[2].sin * q[2].cos * q[1].sin_sq;
    m(1, 2) = -MU2L2SQ * q[1].sin * (q[2].cos_sq - q[2].sin_sq);

    m(2, 0) = m(0, 2);
    m(2, 1) = m(1, 2);
    m(2, 2) = 2.0 * MU2L2SQ * q[2].sin * q[2].cos;
}

void fill_IT_dq4( Eigen::Ref<Eigen::Matrix<double, IT_SIZE, IT_SIZE>> m, std::vector<QData> const& q )
{
    m(0, 0) = 2.0 * MU3 * q[3].value;
    m(1, 1) = m(0, 0);
}

void fill_a( Eigen::Ref<Eigen::Matrix<double, Q_SIZE, Q_SIZE>> m, std::vector<QData> const& q )
{
    m(0, 0) = MU1L1SQ;
    m(1, 1) = MU2L2SQ * q[2].sin_sq;
    m(2, 2) = MU2L2SQ;
    m(3, 3) = MU3;
}

void fill_a_dq1( Eigen::Ref<Eigen::Matrix<double, Q_SIZE, Q_SIZE>> m, std::vector<QData> const& q )
{
    (void)(m);
    (void)(q);
}

void fill_a_dq2( Eigen::Ref<Eigen::Matrix<double, Q_SIZE, Q_SIZE>> m, std::vector<QData> const& q )
{
    (void)(m);
    (void)(q);
}

void fill_a_dq3( Eigen::Ref<Eigen::Matrix<double, Q_SIZE, Q_SIZE>> m, std::vector<QData> const& q )
{
    m(1, 1) = 2.0 * MU2L2SQ * q[2].sin * q[2].cos;
}

void fill_a_dq4( Eigen::Ref<Eigen::Matrix<double, Q_SIZE, Q_SIZE>> m, std::vector<QData> const& q )
{
    (void)(m);
    (void)(q);
}

void fill_A( Eigen::Ref<Eigen::Matrix<double, IT_SIZE, Q_SIZE>> m, std::vector<QData> const& q )
{
    m(0, 1) = - MU2L2SQ * q[1].cos * q[2].sin * q[2].cos;
    
    m(1, 0) = MU1L1SQ;
    m(1, 1) = - MU2L2SQ * q[1].sin * q[2].sin * q[2].cos;
    m(2, 1) = MU2L2SQ * q[2].sin_sq;

    m(0, 2) = - MU2L2SQ * q[1].sin;
    m(1, 2) = MU2L2SQ * q[1].cos;
}

void fill_A_dq1( Eigen::Ref<Eigen::Matrix<double, IT_SIZE, Q_SIZE>> m, std::vector<QData> const& q )
{
    (void)(m);
    (void)(q);
}

void fill_A_dq2( Eigen::Ref<Eigen::Matrix<double, IT_SIZE, Q_SIZE>> m, std::vector<QData> const& q )
{
    m(0, 1) =  MU2L2SQ * q[1].sin * q[2].sin * q[2].cos;
    m(0, 2) = -MU2L2SQ * q[1].cos;
    m(1, 1) = -MU2L2SQ * q[1].cos * q[2].sin * q[2].cos;
    m(1, 2) = -MU2L2SQ * q[1].sin;
}

void fill_A_dq3( Eigen::Ref<Eigen::Matrix<double, IT_SIZE, Q_SIZE>> m, std::vector<QData> const& q )
{
    m(0, 1) = -MU2L2SQ * q[1].cos * (q[2].cos_sq - q[2].sin_sq);
    m(1, 1) = -MU2L2SQ * q[1].sin * (q[2].cos_sq - q[2].sin_sq);
    m(2, 1) = 2 * MU2L2SQ * q[2].sin * q[2].cos;
}

void fill_A_dq4( Eigen::Ref<Eigen::Matrix<double, IT_SIZE, Q_SIZE>> m, std::vector<QData> const& q )
{
    (void)(m);
    (void)(q);
}
