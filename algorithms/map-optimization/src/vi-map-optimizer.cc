#include "map-optimization/vi-map-optimizer.h"

#include <functional>
#include <string>
#include <unordered_map>

#include <map-optimization/callbacks.h>
#include <map-optimization/outlier-rejection-solver.h>
#include <map-optimization/solver-options.h>
#include <map-optimization/solver.h>
#include <map-optimization/vi-optimization-builder.h>
#include <maplab-common/file-logger.h>
#include <maplab-common/progress-bar.h>
#include <visualization/viwls-graph-plotter.h>

#include "Eigen/Core"

DEFINE_int32(
    ba_visualize_every_n_iterations, 3,
    "Update the visualization every n optimization iterations.");

namespace map_optimization {

VIMapOptimizer::VIMapOptimizer(
    visualization::ViwlsGraphRvizPlotter* plotter, bool signal_handler_enabled)
    : plotter_(plotter), signal_handler_enabled_(signal_handler_enabled) {}

bool VIMapOptimizer::optimizeVisualInertial(
    const map_optimization::ViProblemOptions& options,
    const vi_map::MissionIdSet& missions_to_optimize,
    const map_optimization::OutlierRejectionSolverOptions* const
        outlier_rejection_options,
    vi_map::VIMap* map) {
  // outlier_rejection_options is optional.
  CHECK_NOTNULL(map);

  ceres::Solver::Options solver_options =
      map_optimization::initSolverOptionsFromFlags();
  return optimizeVisualInertial(
      options, solver_options, missions_to_optimize, outlier_rejection_options,
      map);
}

bool VIMapOptimizer::optimizeVisualInertial(
    const map_optimization::ViProblemOptions& options,
    const ceres::Solver::Options& solver_options,
    const vi_map::MissionIdSet& missions_to_optimize,
    const map_optimization::OutlierRejectionSolverOptions* const
        outlier_rejection_options,
    vi_map::VIMap* map) {
  // outlier_rejection_options is optional.
  CHECK_NOTNULL(map);

  if (missions_to_optimize.empty()) {
    LOG(WARNING) << "Nothing to optimize.";
    return false;
  }

  map_optimization::OptimizationProblem::UniquePtr optimization_problem(
      map_optimization::constructViProblem(missions_to_optimize, options, map));
  CHECK(optimization_problem != nullptr);

  std::vector<std::shared_ptr<ceres::IterationCallback>> callbacks;
  if (plotter_) {
    map_optimization::appendVisualizationCallbacks(
        FLAGS_ba_visualize_every_n_iterations,
        *(optimization_problem->getOptimizationStateBufferMutable()),
        *plotter_, map, &callbacks);
  }
  map_optimization::appendSignalHandlerCallback(&callbacks);
  ceres::Solver::Options solver_options_with_callbacks = solver_options;
  map_optimization::addCallbacksToSolverOptions(
      callbacks, &solver_options_with_callbacks);

  if (outlier_rejection_options != nullptr) {
    map_optimization::solveWithOutlierRejection(
        solver_options_with_callbacks, *outlier_rejection_options,
        optimization_problem.get());
  } else {
    map_optimization::solve(
        solver_options_with_callbacks, optimization_problem.get());
  }

  if (plotter_ != nullptr) {
    plotter_->visualizeMap(*map);
  }
  return true;
}

bool VIMapOptimizer::optimizeVisualInertial(
    const map_optimization::ViProblemOptions& options,
    const vi_map::MissionIdSet& missions_to_optimize,
    const map_optimization::OutlierRejectionSolverOptions* const
        outlier_rejection_options,
    vi_map::VIMap* map, const pose_graph::VertexId& representative_id,
    const pose_graph::VertexId& next_representative_id,
    double* covariance_next_representative) {
  // outlier_rejection_options is optional.
  CHECK_NOTNULL(map);

  ceres::Solver::Options solver_options =
      map_optimization::initSolverOptionsFromFlags();  // solver-options.h
  return optimizeVisualInertial(
      options, solver_options, missions_to_optimize, outlier_rejection_options,
      map, representative_id, next_representative_id,
      covariance_next_representative);
}

bool VIMapOptimizer::optimizeVisualInertial(
    const map_optimization::ViProblemOptions& options,
    const ceres::Solver::Options& solver_options,
    const vi_map::MissionIdSet& missions_to_optimize,
    const map_optimization::OutlierRejectionSolverOptions* const
        outlier_rejection_options,
    vi_map::VIMap* map, const pose_graph::VertexId& representative_id,
    const pose_graph::VertexId& next_representative_id,
    double* covariance_next_representative) {
  // outlier_rejection_options is optional.
  CHECK_NOTNULL(map);

  if (missions_to_optimize.empty()) {
    LOG(WARNING) << "Nothing to optimize.";
    return false;
  }

  map_optimization::OptimizationProblem::UniquePtr optimization_problem(
      map_optimization::constructViProblem(missions_to_optimize, options, map));
  if (optimization_problem != nullptr) {
    std::cout << "building problem" << std::endl;
  }
  CHECK(optimization_problem != nullptr);

  std::vector<std::shared_ptr<ceres::IterationCallback>> callbacks;
  if (plotter_) {
    map_optimization::appendVisualizationCallbacks(
        FLAGS_ba_visualize_every_n_iterations,
        *(optimization_problem->getOptimizationStateBufferMutable()), *plotter_,
        map, &callbacks);
  }
  map_optimization::appendSignalHandlerCallback(&callbacks);
  ceres::Solver::Options solver_options_with_callbacks = solver_options;
  map_optimization::addCallbacksToSolverOptions(
      callbacks, &solver_options_with_callbacks);

  if (outlier_rejection_options != nullptr) {
    map_optimization::solveWithOutlierRejection(
        solver_options_with_callbacks, *outlier_rejection_options,
        optimization_problem.get());
  } else {
    map_optimization::solve(
        solver_options_with_callbacks, optimization_problem.get());
  }
  // compute covariance of next representative w.r.t. current representative
  ceres::Covariance::Options co_options;
  // options.num_threads = 1;
  // options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  co_options.algorithm_type = ceres::DENSE_SVD;  // Jacobian is difficient
  // co_options.algorithm_type = ceres::SPARSE_QR;
  // options.min_reciprocal_condition_number = 1e-14;  // only for DENSE_SVD
  co_options.null_space_rank = -1;  // only for DENSE_SVD
  // options.apply_loss_function = true;

  ceres::Covariance covariance(co_options);

  std::vector<std::pair<const double*, const double*>> covariance_blocks;
  // vi_map::Vertex& vertex = map->getVertex(next_representative_id);
  // Eigen::Quaterniond q_M_I = vertex.get_q_M_I();
  // Eigen::Matrix<double, 7, 1> vertex_q_IM__M_p_MI;
  // vertex_q_IM__M_p_MI << q_M_I.coeffs(), vertex.get_p_M_I();

  double* vertex_q_IM__M_p_MI_JPL_2 =
      optimization_problem.get()
          ->getOptimizationStateBufferMutable()
          ->get_vertex_q_IM__M_p_MI_JPL(next_representative_id);
  double* vertex_q_IM__M_p_MI_JPL_1 =
      optimization_problem.get()
          ->getOptimizationStateBufferMutable()
          ->get_vertex_q_IM__M_p_MI_JPL(representative_id);
  // get mean of next representative
  std::cout << "Mean: " << std::endl;
  for (int i = 0; i < 7; i++) {
    std::cout << *(vertex_q_IM__M_p_MI_JPL_2 + i) << " ";
  }
  std::cout << std::endl;

  // get covariance of next representative
  covariance_blocks.push_back(
      std::make_pair(vertex_q_IM__M_p_MI_JPL_2, vertex_q_IM__M_p_MI_JPL_2));
  covariance_blocks.push_back(
      std::make_pair(vertex_q_IM__M_p_MI_JPL_1, vertex_q_IM__M_p_MI_JPL_1));
  covariance_blocks.push_back(
      std::make_pair(vertex_q_IM__M_p_MI_JPL_1, vertex_q_IM__M_p_MI_JPL_2));

  double covariance11[7 * 7];
  double covariance12[7 * 7];

  ceres::Problem problem(ceres_error_terms::getDefaultProblemOptions());
  ceres_error_terms::buildCeresProblemFromProblemInformation(
      optimization_problem->getProblemInformationMutable(), &problem);

  CHECK(covariance.Compute(covariance_blocks, &problem));
  covariance.GetCovarianceBlock(
      vertex_q_IM__M_p_MI_JPL_2, vertex_q_IM__M_p_MI_JPL_2,
      covariance_next_representative);
  covariance.GetCovarianceBlock(
      vertex_q_IM__M_p_MI_JPL_1, vertex_q_IM__M_p_MI_JPL_1, covariance11);
  covariance.GetCovarianceBlock(
      vertex_q_IM__M_p_MI_JPL_1, vertex_q_IM__M_p_MI_JPL_2, covariance12);
  std::cout << "Covariance for 11: " << std::endl;
  for (int j = 0; j < 49; j++) {
    std::cout << *(covariance11 + j) << " ";
  }
  std::cout << std::endl;

  std::cout << "Covariance for 22: " << std::endl;
  for (int j = 0; j < 49; j++) {
    std::cout << *(covariance_next_representative + j) << " ";
  }
  std::cout << std::endl;

  Eigen::Map<Eigen::MatrixXd> Covariance22(
      covariance_next_representative, 7, 7);
  Eigen::MatrixXd Covariance11 =
      Eigen::Map<Eigen::MatrixXd>(covariance11, 7, 7);
  Eigen::MatrixXd Covariance12 =
      Eigen::Map<Eigen::MatrixXd>(covariance12, 7, 7);
  Eigen::MatrixXd Covariance11_ = Covariance11.block<4, 4>(0, 0);
  Eigen::MatrixXd Covariance12_ = Covariance12.block<4, 7>(0, 0);
  // Covariance22 = Covariance22 - Covariance12_.transpose() *
  // Covariance11_.inverse() * Covariance12_;
  std::cout << "Covariance: " << std::endl;
  for (int j = 0; j < 49; j++) {
    std::cout << *(covariance_next_representative + j) << " ";
  }
  std::cout << std::endl;

  if (plotter_ != nullptr) {
    plotter_->visualizeMap(*map);
  }
  return true;
}
bool VIMapOptimizer::optimizeVisualInertial(
    const map_optimization::ViProblemOptions& options,
    const vi_map::MissionIdSet& missions_to_optimize,
    const map_optimization::OutlierRejectionSolverOptions* const
        outlier_rejection_options,
    vi_map::VIMap* map, const pose_graph::VertexId& next_representative_id,
    const vi_map::MissionId& new_mission) {
  CHECK_NOTNULL(map);
  ceres::Solver::Options solver_options =
      map_optimization::initSolverOptionsFromFlags();  // solver-options.h
  return optimizeVisualInertial(
      options, solver_options, missions_to_optimize, outlier_rejection_options,
      map, next_representative_id, new_mission);
}

bool VIMapOptimizer::optimizeVisualInertial(
    const map_optimization::ViProblemOptions& options,
    const ceres::Solver::Options& solver_options,
    const vi_map::MissionIdSet& missions_to_optimize,
    const map_optimization::OutlierRejectionSolverOptions* const
        outlier_rejection_options,
    vi_map::VIMap* map, const pose_graph::VertexId& next_representative_id,
    const vi_map::MissionId& new_mission) {
  CHECK_NOTNULL(map);

  if (missions_to_optimize.empty()) {
    LOG(WARNING) << "Nothing to optimize.";
    return false;
  }

  map_optimization::OptimizationProblem::UniquePtr optimization_problem(
      map_optimization::constructViProblem_submap(
          missions_to_optimize, options, map));
  optimization_problem->applyGaugeFixesForGivenVertex(
      next_representative_id, new_mission);
  CHECK(optimization_problem != nullptr);

  std::vector<std::shared_ptr<ceres::IterationCallback>> callbacks;
  if (plotter_) {
    map_optimization::appendVisualizationCallbacks(
        FLAGS_ba_visualize_every_n_iterations,
        *(optimization_problem->getOptimizationStateBufferMutable()), *plotter_,
        map, &callbacks);
  }
  map_optimization::appendSignalHandlerCallback(&callbacks);
  ceres::Solver::Options solver_options_with_callbacks = solver_options;
  map_optimization::addCallbacksToSolverOptions(
      callbacks, &solver_options_with_callbacks);

  if (outlier_rejection_options != nullptr) {
    map_optimization::solveWithOutlierRejection(
        solver_options_with_callbacks, *outlier_rejection_options,
        optimization_problem.get());
  } else {
    map_optimization::solve(
        solver_options_with_callbacks, optimization_problem.get());
  }

  if (plotter_ != nullptr) {
    plotter_->visualizeMap(*map);
  }
  return true;
}

}  // namespace map_optimization
