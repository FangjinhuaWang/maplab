#ifndef HIERARCHICAL_BUNDLE_ADJUSTMENT_POSEGRAPH_ERROR_H_
#define HIERARCHICAL_BUNDLE_ADJUSTMENT_POSEGRAPH_ERROR_H_

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "ceres/autodiff_cost_function.h"
#include <istream>
#include <map>
#include <string>
#include <vector>

#include "vi-map/vi-map.h"
#include "vi-map/vertex.h"


namespace posegraph_error {
struct Pose3d {
Eigen::Vector3d p;
Eigen::Quaterniond q;

EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
// map <pose_graph::VertexId, Pose3d>
typedef std::map<int, Pose3d, std::less<int>,
                 Eigen::aligned_allocator<std::pair<const int, Pose3d> > >
    MapOfPoses;

struct Constraint3d{
  int id_begin_;
  int id_end_;

  Pose3d transformation_;

  Eigen::Matrix<double, 6, 6> information_;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
// vector<Constraint3d>
typedef std::vector<Constraint3d, Eigen::aligned_allocator<Constraint3d> >
    VectorOfConstraints;

class PoseGraphError{
 public:
  PoseGraphError(const Pose3d& t_ab_measured,
                const Eigen::Matrix<double, 6, 6>& sqrt_information)
      : t_ab_measured_(t_ab_measured), sqrt_information_(sqrt_information) {}

  template <typename T>
  bool operator()(const T* const p_a_ptr, const T* const q_a_ptr,
                  const T* const p_b_ptr, const T* const q_b_ptr,
                  T* residuals_ptr) const {
  	// position and rotation for first vertex
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > p_a(p_a_ptr);
    Eigen::Map<const Eigen::Quaternion<T> > q_a(q_a_ptr);
    // position and rotation for second vertex
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > p_b(p_b_ptr);
    Eigen::Map<const Eigen::Quaternion<T> > q_b(q_b_ptr);

    // Compute the relative transformation between the two frames.
    Eigen::Quaternion<T> q_a_inverse = q_a.conjugate();
    Eigen::Quaternion<T> q_ab_estimated = q_a_inverse * q_b;
    // Eigen::Quaternion<T> q_b_inverse = q_b.conjugate();
    // Eigen::Quaternion<T> q_ab_estimated = q_b_inverse * q_a;

    // Represent the displacement between the two frames in the A frame.
    Eigen::Matrix<T, 3, 1> p_ab_estimated = q_a_inverse * (p_b - p_a);

    // Compute the error between the two orientation estimates.
    Eigen::Quaternion<T> delta_q =
        t_ab_measured_.q.template cast<T>() * q_ab_estimated.conjugate();

    // Compute the residuals.
    // [ position         ]   [ delta_p          ]
    // [ orientation (3x1)] = [ 2 * delta_q(0:2) ]
    Eigen::Map<Eigen::Matrix<T, 6, 1> > residuals(residuals_ptr);
    residuals.template block<3, 1>(0, 0) =
        p_ab_estimated - t_ab_measured_.p.template cast<T>();
    residuals.template block<3, 1>(3, 0) = T(2.0)* delta_q.vec();

    // Scale the residuals by the measurement uncertainty.
    residuals.applyOnTheLeft(sqrt_information_.template cast<T>());

    return true;
  }

  static ceres::CostFunction* Create(
      const Pose3d& t_ab_measured,
      const Eigen::Matrix<double, 6, 6>& sqrt_information) {
    return new ceres::AutoDiffCostFunction<PoseGraphError, 6, 3, 4, 3, 4>(
        new PoseGraphError(t_ab_measured, sqrt_information));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  // The measurement for the position of B relative to A in the A frame.
  const Pose3d t_ab_measured_;
  // The square root of the measurement information matrix.
  const Eigen::Matrix<double, 6, 6> sqrt_information_;
};

void getConstraints(
  VectorOfConstraints* constraints,
  const int id_begin_,
  const int id_end_,
  const Pose3d transformation_,
  const Eigen::Matrix<double, 6, 6> information_);

void BuildOptimizationProblem(const VectorOfConstraints& constraints,
                              MapOfPoses* poses, ceres::Problem* problem);

bool SolveOptimizationProblem(ceres::Problem* problem);

}	// namespace posegraph_error

#endif	// HIERARCHICAL_BUNDLE_ADJUSTMENT_POSEGRAPH_ERROR_H_