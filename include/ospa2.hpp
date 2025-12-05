#pragma once

#include <vector>
#include <cstddef>
#include <tuple>
#include <Eigen/Dense>

using Eigen::Map;
using Eigen::VectorXd;

// // Compact block representing a trajectory: contiguous row-major flat array of length (L*D)
// struct TrajBlock {
//     const double* data = nullptr; // pointer to L*D doubles
//     size_t length = 0; // number of frames (L)
//     size_t dim = 0; // dimension per frame (D)
// };

// struct OSPA2Params {
//     double c = 100.0;
//     double p = 1.0;
//     double q = 1.0;
// };

// // compute pairwise distance between two trajectory blocks (ND). returns scalar distance
// double traj_distance(const TrajBlock &a, const TrajBlock &b, double c, double q);

// compute full distance matrix (m x n) between gt and trk
// std::vector<std::vector<double>> compute_distance_matrix(
//     const std::vector<TrajBlock>& gt,
//     const std::vector<TrajBlock>& trk,
//     double c, double q);

// // Given a cost matrix D, compute OSPA2, loc and card using an assignment algorithm.
// std::tuple<double,double,double> ospa2_from_matrix(
//     const std::vector<std::vector<double>>& D, double c, double p);


std::tuple<double,double,double> ospa2_from_matrix_bind(
    const Eigen::MatrixXd& D, double c, double p);



Eigen::MatrixXd compute_distance_matrix_bind(
    const std::vector<Eigen::MatrixXd>& gt,
    const std::vector<Eigen::MatrixXd>& trk,
    double c, double q);