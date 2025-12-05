#pragma once
#include <vector>
#include <tuple>
#include "lap.h"   // LAPJV AVX2 https://github.com/src-d/lapjv/tree/master

// inline std::vector<int> lapjv_solve(const std::vector<std::vector<double>>& cost)
// {
//     size_t m = cost.size();
//     size_t n = m ? cost[0].size() : 0;
//     size_t dim = std::max(m, n);

//     std::vector<double> assign_cost(dim * dim, 0.0);

//     for (size_t i = 0; i < dim; i++) {
//         for (size_t j = 0; j < dim; j++) {
//             if (i < m && j < n)
//                 assign_cost[i * dim + j] = cost[i][j];
//             else
//                 assign_cost[i * dim + j] = 1e9; 
//         }
//     }

//     std::vector<int> rowsol(dim), colsol(dim);
//     std::vector<double> u(dim), v(dim);

//     lap<true /*AVX2*/, false /*verbose*/, int, double>(
//         (int)dim,
//         assign_cost.data(),
//         rowsol.data(),
//         colsol.data(),
//         u.data(),
//         v.data()
//     );

//     rowsol.resize(m);
//     return rowsol;
// }


inline std::vector<int> lapjv_solve_bis(const Eigen::MatrixXd& cost)
{
    size_t m = cost.rows();
    size_t n = cost.cols();
    size_t dim = std::max(m, n);

    std::vector<double> assign_cost(dim * dim, 0.0);

    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            if (i < m && j < n)
                // assign_cost[i * dim + j] = cost[i][j];
                assign_cost[i * dim + j] = cost(i,j);

            else
                assign_cost[i * dim + j] = 1e9; 
        }
    }

    std::vector<int> rowsol(dim), colsol(dim);
    std::vector<double> u(dim), v(dim);

    lap<true /*AVX2*/, false /*verbose*/, int, double>(
        (int)dim,
        assign_cost.data(),
        rowsol.data(),
        colsol.data(),
        u.data(),
        v.data()
    );

    rowsol.resize(m);
    return rowsol;
}