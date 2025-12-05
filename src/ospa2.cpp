#include "ospa2.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <omp.h>
// #include <Eigen/Dense>
#include "lap_wrapper.h"

// using Eigen::Map;
// using Eigen::VectorXd;

// Compute ND Euclidean distance for frame k (assumes contiguous row-major L x D)
// static inline double frame_distance_nd(const double* a_ptr, const double* b_ptr, size_t dim) {
//     // Use Eigen Map for vectorized dot
//     Map<const VectorXd> va(a_ptr, static_cast<Eigen::Index>(dim));
//     Map<const VectorXd> vb(b_ptr, static_cast<Eigen::Index>(dim));
//     double dist2 = (va - vb).squaredNorm();
//     return std::sqrt(dist2);
// }

// // Distance between two trajectories
// double traj_distance(const TrajBlock &a, const TrajBlock &b, double c, double q) {
//     if (a.dim == 0 || b.dim == 0) throw std::invalid_argument("trajectory dimension is zero");
//     if (a.dim != b.dim) throw std::invalid_argument("trajectory dimension mismatch");

//     size_t L = std::max(a.length, b.length);
//     if (L == 0) return 0.0;
//     size_t D = a.dim;

//     double sum = 0.0;
//     for (size_t k = 0; k < L; ++k) {
//         double dd;
//         if (k < a.length && k < b.length) {
//             const double* ap = a.data + k*D;
//             const double* bp = b.data + k*D;
//             double dist = frame_distance_nd(ap, bp, D);
//             dd = std::min(dist, c);
//         } else {
//             dd = c; // penalty for missing frame
//         }
//         double term = std::pow(dd / static_cast<double>(L), q);
//         sum += term;
//     }
//     return std::pow(sum, 1.0 / q);
// }

// // compute distance matrix with OpenMP parallelization
// std::vector<std::vector<double>> compute_distance_matrix(
//     const std::vector<TrajBlock>& gt,
//     const std::vector<TrajBlock>& trk,
//     double c, double q)
// {
//     size_t m = gt.size();
//     size_t n = trk.size();
//     std::vector<std::vector<double>> D(m, std::vector<double>(n));

//     #pragma omp parallel for schedule(dynamic)
//     for (size_t i = 0; i < m; ++i) {
//         for (size_t j = 0; j < n; ++j) {
//             D[i][j] = traj_distance(gt[i], trk[j], c, q);
//         }
//     }
//     return D;
// }


//te OSPA2 distance between two sets of trajectories
Eigen::MatrixXd compute_distance_matrix_bind(
    const std::vector<Eigen::MatrixXd>& gt,
    const std::vector<Eigen::MatrixXd>& trk,
    double c, double q)
{
    int m = gt.size();
    int n = trk.size();
    Eigen::MatrixXd D(m, n);
    D.setZero();
    #pragma omp parallel for if(m*n>100) 
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            int L = std::max(gt[i].rows(), trk[j].rows());
            int dim = gt[i].cols();
            if (trk[j].cols() != dim) throw std::runtime_error("All trajectories must have same dimension D");

            for (int k = 0; k < L; ++k) {
                double d = c;
                if (k < gt[i].rows() && k < trk[j].rows()){
                    Eigen::RowVectorXd g_row, t_row;
                    g_row = gt[i].row(k);
                    t_row = trk[j].row(k);

                    d = (g_row - t_row).norm();
                    if (d > c) d = c;
                }
                sum += std::pow(d *(1.0/L), q);
            }
            D(i,j) = std::pow(sum, 1.0 / q);
        }
    }
    return D;
}


// std::tuple<double,double,double> ospa2_from_matrix(
//     const std::vector<std::vector<double>>& D, double c, double p)
// {
//     size_t m = D.size();
//     size_t n = m ? D[0].size() : 0;

//     if (m==0 && n==0)
//         return {0.0, 0.0, 0.0};

//     if (m==0 || n==0) {
//         double card = std::pow(std::pow(c,p) * std::abs((int)m - (int)n) / (double)std::max(m,n), 1.0/p);
//         return {card, 0.0, card};
//     }

//     std::vector<int> assign = lapjv_solve(D);

//     double spatial_cost = 0.0;
//     for (size_t i = 0; i < m; ++i) {
//         int j = assign[i];
//         if (j >= 0 && (size_t)j < n) {
//             spatial_cost += D[i][j];
//         }
//     }

//     double loc = std::pow(spatial_cost / (double)std::max(m,n), 1.0/p);
//     double card = std::pow(std::pow(c,p) * std::abs((int)m - (int)n) / (double)std::max(m,n), 1.0/p);
//     double ospa2 = std::pow((spatial_cost + std::pow(c,p) * std::abs((int)m - (int)n)) / (double)std::max(m,n), 1.0/p);

//     return {ospa2, loc, card};
// }


std::tuple<double,double,double> ospa2_from_matrix_bind(
    const Eigen::MatrixXd& D, double c, double p)
{
    size_t m = D.rows();
    size_t n = D.cols();
    if (m==0 && n==0)
        return {0.0, 0.0, 0.0};

    if (m==0 || n==0) {
        double card = std::pow(std::pow(c,p) * std::abs((int)m - (int)n) / (double)std::max(m,n), 1.0/p);
        return {card, 0.0, card};
    }

    std::vector<int> assign = lapjv_solve_bis(D);

    double spatial_cost = 0.0;
    for (size_t i = 0; i < m; ++i) {
        int j = assign[i];
        if (j >= 0 && (size_t)j < n) {
            // spatial_cost += D[i][j];
            spatial_cost += D(i,j);

        }
    }

    double loc = std::pow(spatial_cost / (double)std::max(m,n), 1.0/p);
    double card = std::pow(std::pow(c,p) * std::abs((int)m - (int)n) / (double)std::max(m,n), 1.0/p);
    double ospa2 = std::pow((spatial_cost + std::pow(c,p) * std::abs((int)m - (int)n)) / (double)std::max(m,n), 1.0/p);

    return {ospa2, loc, card};
}