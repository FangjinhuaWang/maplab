#include <iostream>
#include <fstream>
#include <string>

#include "ceres/ceres.h"

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "hierarchical-bundle-adjustment-plugin/posegraph_error.h"




namespace posegraph_error {



void getConstraints(
  VectorOfConstraints* constraints,
  const int id_begin_,
  const int id_end_,
  const Pose3d transformation_,
  const Eigen::Matrix<double, 6, 6> information_){
  // CHECK(constraints != NULL);
  // constraints->clear();
  Constraint3d constraint;
  constraint.id_begin_ = id_begin_;
  constraint.id_end_ = id_end_;
  constraint.transformation_ = transformation_;
  constraint.information_ = information_;
  constraints->push_back(constraint);
}

// Constructs the nonlinear least squares optimization problem from the pose
// graph constraints.
void BuildOptimizationProblem(const VectorOfConstraints& constraints,
                              MapOfPoses* poses, ceres::Problem* problem) {
  CHECK(poses != NULL);
  CHECK(problem != NULL);
  if (constraints.empty()) {
    LOG(INFO) << "No constraints, no problem to optimize.";
    return;
  }

  ceres::LossFunction* loss_function = NULL;
  ceres::LocalParameterization* quaternion_local_parameterization =
      new EigenQuaternionParameterization;

  for (VectorOfConstraints::const_iterator constraints_iter =
           constraints.begin();
       constraints_iter != constraints.end(); ++constraints_iter) {
    const Constraint3d& constraint = *constraints_iter;

    MapOfPoses::iterator pose_begin_iter = poses->find(constraint.id_begin_);

    MapOfPoses::iterator pose_end_iter = poses->find(constraint.id_end_);

    const Eigen::Matrix<double, 6, 6> sqrt_information =
        constraint.information_.llt().matrixL();
    // Ceres will take ownership of the pointer.
    ceres::CostFunction* cost_function =
        PoseGraphError::Create(constraint.transformation_, sqrt_information);

    problem->AddResidualBlock(cost_function, loss_function,
                              pose_begin_iter->second.p.data(),
                              pose_begin_iter->second.q.coeffs().data(),
                              pose_end_iter->second.p.data(),
                              pose_end_iter->second.q.coeffs().data());

    problem->SetParameterization(pose_begin_iter->second.q.coeffs().data(),
                                 quaternion_local_parameterization);
    problem->SetParameterization(pose_end_iter->second.q.coeffs().data(),
                                 quaternion_local_parameterization);
  }

  MapOfPoses::iterator pose_start_iter = poses->begin();
  
  problem->SetParameterBlockConstant(pose_start_iter->second.p.data());
  problem->SetParameterBlockConstant(pose_start_iter->second.q.coeffs().data());
}

// Returns true if the solve was successful.
bool SolveOptimizationProblem(ceres::Problem* problem) {
  CHECK(problem != NULL);

  ceres::Solver::Options options;
  options.max_num_iterations = 30;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  ceres::Solver::Summary summary;
  ceres::Solve(options, problem, &summary);

  std::cout << summary.FullReport() << '\n';

  return summary.IsSolutionUsable();
}


}  // namespace posegraph_error