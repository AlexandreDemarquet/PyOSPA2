#include "ospa2.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <omp.h>
#include "lap_wrapper.h"



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