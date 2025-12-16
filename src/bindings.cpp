#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include "ospa2.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// // Compute OSPA2 distance between two sets of trajectories
// Eigen::MatrixXd compute_distance_matrix_bind(
//     const std::vector<Eigen::MatrixXd>& gt,
//     const std::vector<Eigen::MatrixXd>& trk,
//     double c, double q)
// {
//     int m = gt.size();
//     int n = trk.size();
//     Eigen::MatrixXd D(m, n);
//     D.setZero();
//     #pragma omp parallel for if(m*n>100) 
//     for (int i = 0; i < m; ++i) {
//         for (int j = 0; j < n; ++j) {
//             double sum = 0.0;
//             int L = std::max(gt[i].rows(), trk[j].rows());
//             int dim = gt[i].cols();
//             if (trk[j].cols() != dim) throw std::runtime_error("All trajectories must have same dimension D");

//             for (int k = 0; k < L; ++k) {
//                 double d = c;
//                 if (k < gt[i].rows() && k < trk[j].rows()){
//                     Eigen::RowVectorXd g_row, t_row;
//                     g_row = gt[i].row(k);
//                     t_row = trk[j].row(k);

//                     d = (g_row - t_row).norm();
//                     if (d > c) d = c;
//                 }
//                 sum += std::pow(d *(1.0/L), q);
//             }
//             D(i,j) = std::pow(sum, 1.0 / q);
//         }
//     }
//     return D;
// }

// // Compute OSPA2 score from distance matrix
// py::tuple ospa2_from_matrix_bind(const Eigen::MatrixXd& D, double c, double p)
// {
//     size_t m = D.rows();
//     size_t n = D.cols();
//     std::vector<std::vector<double>> Dvec(m, std::vector<double>(n, 0.0));
//     for (size_t i = 0; i < m; ++i)
//         for (size_t j = 0; j < n; ++j)
//             Dvec[i][j] = D(i,j);

//     auto tup = ospa2_from_matrix(Dvec, c, p);
//     return py::make_tuple(std::get<0>(tup), std::get<1>(tup), std::get<2>(tup));
// }

PYBIND11_MODULE(_ospa2, m) {
    m.doc() = "_ospa2: fast OSPA2 C++ backend using Eigen and OpenMP";

    m.def("compute_distance_matrix", &compute_distance_matrix_bind,
          py::arg("gt"), py::arg("trk"), py::arg("c")=100.0, py::arg("q")=1.0,
          R"pbdoc(
            compute_distance_matrix(gt, trk, c=100.0, q=1.0) -> 2D list
            gt and trk: list of Eigen::MatrixXd (L,D)
            Returns the OSPA2 distance matrix
          )pbdoc");

    m.def("ospa2_from_matrix", &ospa2_from_matrix_bind,
          py::arg("D"), py::arg("c")=100.0, py::arg("p")=1.0,
          R"pbdoc(
            ospa2_from_matrix(D, c=100.0, p=1.0) -> (ospa2, loc, card)
            D: Eigen::MatrixXd (distance matrix)
          )pbdoc");

    m.def("average_ospa2_at_time", &average_ospa2_at_time_cpp,
          py::arg("pairs"), py::arg("c")=100.0, py::arg("p")=1.0, py::arg("q")=1.0,
          R"pbdoc(
            average_ospa2_at_time(pairs, c=100.0, p=1.0, q=1.0) -> (avg_ospa2, avg_loc, avg_card)
            pairs: list of pairs (gt_trajs, trk_trajs) where each is list of Eigen::MatrixXd
            Averages OSPA2 over multiple pairs in parallel using OpenMP
          )pbdoc");
}
