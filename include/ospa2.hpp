#pragma once

#include <vector>
#include <cstddef>
#include <tuple>
#include <Eigen/Dense>

using Eigen::Map;
using Eigen::VectorXd;


std::tuple<double,double,double> ospa2_from_matrix_bind(
    const Eigen::MatrixXd& D, double c, double p);


Eigen::MatrixXd compute_distance_matrix_bind(
    const std::vector<Eigen::MatrixXd>& gt,
    const std::vector<Eigen::MatrixXd>& trk,
    double c, double q);