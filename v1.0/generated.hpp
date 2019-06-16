#pragma once

#include <vector>

#include <Eigen/Dense>

#include "constants.hpp"
#include "masses.hpp"

#include "QData.hpp"

const double MU1 = masses::O16_AMU / 2.0; 
const double MU2 = masses::CO2_AMU * masses::AR40_AMU / (masses::CO2_AMU + masses::AR40_AMU);
const double L2 = lengths::L_CO2_ALU * lengths::L_CO2_ALU; 

void fill_IT( Eigen::Ref<Eigen::Matrix<double, IT_SIZE, IT_SIZE>> m, std::vector<QData> const& q )
{
    m(0, 0) = MU1 * L2 * q[1].cos_sq + MU2 * q[0].sq;
    m(2, 0) = - MU1 * L2 * q[1].sin * q[1].cos;
    m(0, 2) = m(2, 0);
    m(2, 2) = MU1 * L2 * q[1].sin * q[1].sin;
    m(1, 1) = m(0, 0) + m(2, 2); 
}

void fill_IT_dR( Eigen::Ref<Eigen::Matrix<double, IT_SIZE, IT_SIZE>> m, std::vector<QData> const& q )
{
	m(0, 0) = 2.0 * MU2 * q[0].value;
	m(1, 1) = 2.0 * MU2 * q[0].value;
}

void fill_IT_dTheta( Eigen::Ref<Eigen::Matrix<double, IT_SIZE, IT_SIZE>> m, std::vector<QData> const& q )
{
	m(0, 0) = - 2.0 * MU1 * L2 * q[1].sin * q[1].cos;
	m(2, 0) = - MU1 * L2 * (q[1].cos_sq - q[1].sin_sq);
	m(0, 2) = m(2, 0);
	m(2, 2) = -m(0, 0);
}

void fill_a( Eigen::Ref<Eigen::Matrix<double, Q_SIZE, Q_SIZE>> m, std::vector<QData> const& q )
{
	(void)(q);
	m(0, 0) = MU2;
	m(1, 1) = MU1 * L2;
}

void fill_a_dR( Eigen::Ref<Eigen::Matrix<double, Q_SIZE, Q_SIZE>> m, std::vector<QData> const& q )
{
	(void)(m);
	(void)(q);
}

void fill_a_dTheta( Eigen::Ref<Eigen::Matrix<double, Q_SIZE, Q_SIZE>> m, std::vector<QData> const& q )
{
	(void)(m);
	(void)(q);
}

void fill_A( Eigen::Ref<Eigen::Matrix<double, IT_SIZE, Q_SIZE>> m, std::vector<QData> const& q )
{
	(void)(q);
	m(1, 1) = MU1 * L2;
}

void fill_A_dR( Eigen::Ref<Eigen::Matrix<double, IT_SIZE, Q_SIZE>> m, std::vector<QData> const& q )
{
	(void)(m);
	(void)(q);
}

void fill_A_dTheta( Eigen::Ref<Eigen::Matrix<double, IT_SIZE, Q_SIZE>> m, std::vector<QData> const& q )
{
	(void)(m);
	(void)(q);
}




