#include "hierarchical-bundle-adjustment-plugin/optimizer-plugin.h"
#include <math.h>
#include <time.h>
#include <ceres/ceres.h>
#include <console-common/basic-console-plugin.h>
#include <console-common/console.h>
#include <map-manager/map-manager.h>
#include <map-optimization/outlier-rejection-solver.h>
#include <map-optimization/vi-optimization-builder.h>
#include <vi-map/vi-map.h>
#include <vi-map/check-map-consistency.h>
#include <visualization/viwls-graph-plotter.h>
// first use metics to partition the graph
#include <vi-map-helpers/vi-map-partitioner.h>

#include "map-optimization/vi-map-optimizer.h"
#include "map-optimization/vi-map-relaxation.h"

#include "hierarchical-bundle-adjustment-plugin/posegraph_error.h"

DEFINE_bool(
    ba_use_outlier_rejection_solver_, true,
    "Reject outlier landmarks during the solve?");
DECLARE_string(map_mission);
DECLARE_string(map_mission_list);

DEFINE_bool(partition, true, 
    "partition the map into submaps");
DEFINE_bool(optimize_submaps, true, 
    "optimize each submap to get constraints to add in the higher level map");
DEFINE_bool(optimize_posegraph, true,
    "optimize the posegraph in the higher level map");
DEFINE_bool(optimize_submaps_second, true,
    "apply rigid transformation in each submap, fix representatives and then optimize each submap");
DEFINE_bool(fix_rotation_all, true,
    "fix all of rotation DoFs of the root vertex of each submap when perform optimization");

namespace hierarchical_bundle_djustment_plugin {
OptimizerPlugin::OptimizerPlugin(
    common::Console* console, visualization::ViwlsGraphRvizPlotter* plotter)
    : common::ConsolePluginBaseWithPlotter(console, plotter) {
  addCommand(
      {"hi_optimize_visual", "hi_optv"},
      [this]() -> int {
        constexpr bool kVisualOnly = true;
        return optimizeVisualInertial_naive(
            kVisualOnly, FLAGS_ba_use_outlier_rejection_solver_);
      },
      "Visual optimization over the selected missions "
      "(per default all).",
      common::Processing::Sync);
  addCommand(
      {"hi_optimize_visual_inertial", "hi_optvi"},
      [this]() -> int {
        constexpr bool kVisualOnly = false;
        return optimizeVisualInertial_naive(
            kVisualOnly, FLAGS_ba_use_outlier_rejection_solver_);
      },
      "Visual-inertial optimization over the selected missions "
      "(per default all).",
      common::Processing::Sync);
  addCommand(
      {"hi_relax"}, [this]() -> int { return relaxMap(); }, "nRelax posegraph.",
      common::Processing::Sync);
  addCommand(
      {"hi_relax_missions_independently", "hi_srelax"},
      [this]() -> int { return relaxMapMissionsSeparately(); },
      "Relax missions separately.", common::Processing::Sync);
}


int OptimizerPlugin::optimizeVisualInertial_naive(
    bool visual_only, bool outlier_rejection) {
  // Select map and missions to optimize.
  // partition the map in such way:
  // only one mission in the map
  // partition the vertices along the trajectory into missions
  std::string selected_map_key;
  if (!getSelectedMapKeyIfSet(&selected_map_key)) {
    return common::kStupidUserError;
  }
  vi_map::VIMapManager map_manager;
  vi_map::VIMapManager::MapWriteAccess map =
      map_manager.getMapWriteAccess(selected_map_key);

  vi_map::MissionIdList missions_to_optimize_list;
  if (!FLAGS_map_mission.empty()) {
    if (!FLAGS_map_mission_list.empty()) {
      LOG(ERROR) << "Please provide only one of --map_mission and "
                 << "--map_mission_list.";
      return common::kStupidUserError;
    }
    vi_map::MissionId mission_id;
    if (!map->hexStringToMissionIdIfValid(FLAGS_map_mission, &mission_id)) {
      LOG(ERROR) << "The given mission id \"" << FLAGS_map_mission
                 << "\" is not valid.";
      return common::kStupidUserError;
    }
    missions_to_optimize_list.emplace_back(mission_id);
  } else if (!FLAGS_map_mission_list.empty()) {
    if (!vi_map::csvIdStringToIdList(
            FLAGS_map_mission_list, &missions_to_optimize_list)) {
      LOG(ERROR) << "The provided CSV mission id list is not valid!";
      return common::kStupidUserError;
    }
  } else {
    map->getAllMissionIds(&missions_to_optimize_list);
  }
  vi_map::MissionIdSet missions_to_optimize(
      missions_to_optimize_list.begin(), missions_to_optimize_list.end());

  map_optimization::ViProblemOptions options =
      map_optimization::ViProblemOptions::initFromGFlags();
  if (visual_only) {
    options.add_inertial_constraints = false;
  }
  // ------------------------------------------------------
  // revise this part. replace map with submaps
  
  time_t start_time;
  start_time = std::time(NULL);
  // number of vertices in each segment(submap)
  const unsigned int num_of_vertex_per_submap = 10;
  std::vector<pose_graph::VertexIdList> partitioning; 

  // partition the map into different segments
  partitionMapinTimeOrder(*(map.get()), num_of_vertex_per_submap, &partitioning);
  
  vi_map::MissionIdList original_mission_ids;
  original_mission_ids.clear();
  // only for one mission case
  (map.get())->getAllMissionIds(&original_mission_ids);
  // -----------------------------------------------------------
  // // duplicate original missions
  // vi_map::MissionIdList duplicate_mission_ids;
  // duplicate_mission_ids.clear();
  // for (const vi_map::MissionId& original_id : original_mission_ids){
  //   (map.get())->duplicateMission(original_id);
  // }
  // vi_map::MissionIdList combined_mission_ids;
  // combined_mission_ids.clear();
  // (map.get())->getAllMissionIds(&combined_mission_ids);

  // for (const vi_map::MissionId& mission_id : combined_mission_ids){
  //   if (std::find(original_mission_ids.begin(),original_mission_ids.end(),
  //     mission_id) == original_mission_ids.end()){
  //     duplicate_mission_ids.insert(duplicate_mission_ids.end(), mission_id);
  //   }
  // }
  // -------------------------------------------------------------

  // store the id for representatives and their copies <original, copy>
  std::unordered_map<pose_graph::VertexId, pose_graph::VertexId> source_to_dest_vertex_id_map;
  source_to_dest_vertex_id_map.clear();

  
  // record poses of all representatives for optimization
  // in the higher level map
  std::unordered_map<pose_graph::VertexId, int> vertex_id_to_index;
  vertex_id_to_index.clear();
  typedef std::unordered_map<pose_graph::VertexId, int>::iterator vertex_id_index;
  posegraph_error::MapOfPoses map_of_poses;
  map_of_poses.clear();
  int kkk = 0;
  for (const pose_graph::VertexIdList& partition: partitioning){
    const pose_graph::VertexId& vertex_id = partition.front();
    const vi_map::Vertex& vertex = (map.get())->getVertex(vertex_id);
    posegraph_error::Pose3d pose;
    pose.p = vertex.get_p_M_I();
    pose.q = vertex.get_q_M_I().normalized();
    map_of_poses.emplace(kkk, pose);
    vertex_id_to_index.emplace(vertex_id, kkk);
    kkk++;
  }
  vertex_id_index vertexid2index;

  // copy the representatives from 2nd submap (partition)
  pose_graph::VertexIdList representative_ids;
  unsigned int k = 0u; 
  representative_ids.clear();
  for (const pose_graph::VertexIdList& partition: partitioning){
    if(k>0){
      pose_graph::VertexId representative_vertex_id = partition.front();
      representative_ids.push_back(representative_vertex_id);
    }
    k++;
  }

  (*(map.get())).clone_representative(representative_ids,
                                      &source_to_dest_vertex_id_map);

  // make sure that the map is consistent
  (*(map.get())).ensure_consistency(partitioning,
                                    source_to_dest_vertex_id_map);
  // build missions for submaps (partitions)
  // also delete the old one
  vi_map::MissionIdList new_missions;
  new_missions.clear();
  (*(map.get())).build_missions_from_partition(partitioning,
                                    source_to_dest_vertex_id_map,
                                    &new_missions,
                                    original_mission_ids);
  // std::cout << "mission is done" << std::endl;
  // std::cout << new_missions.size() << std::endl;
  // (map.get())->getAllMissionIds(&original_mission_ids);
  
  // check map consistency
  bool consistent = vi_map::checkMapConsistency(*(map.get()));
  std::cout << "consistency" << consistent << std::endl;

  if (!FLAGS_optimize_submaps){
    return common::kSuccess;
  }
  // ------------do optimization for submaps (partitions)-----------------
  // -------------add constraints in the higher level map-----------------
  posegraph_error::VectorOfConstraints constraints;
  constraints.clear();
  unsigned int kk = 0u;
  typedef std::unordered_map<pose_graph::VertexId, pose_graph::VertexId>::iterator
                VertexIdToVertexIdIterator;
  for(const vi_map::MissionId& new_mission : new_missions){
    // optimize the submaps and get the mean and covariance
    if(new_mission != new_missions.back()){
      
      pose_graph::VertexId representative_id = partitioning[kk].front();
      pose_graph::VertexId next_representative_id = partitioning[kk+1].front();
      VertexIdToVertexIdIterator it;
      it = source_to_dest_vertex_id_map.find(next_representative_id);
      pose_graph::VertexId copy_next_representative_id = it->second;
      // compute covariance of next representative w.r.t. current representative
      double covariance[7*7];
      // double covariance[7*7] = {1,0,0,0,0,0,0,
      //                           0,1,0,0,0,0,0,
      //                           0,0,1,0,0,0,0,
      //                           0,0,0,1,0,0,0,
      //                           0,0,0,0,1,0,0,
      //                           0,0,0,0,0,1,0,
      //                           0,0,0,0,0,0,1}; // rotation:4 translation:3
      
      

      vi_map::MissionIdSet missions_to_optimize;
      missions_to_optimize.clear();

      missions_to_optimize.insert(new_mission);
      map_optimization::VIMapOptimizer optimizer(plotter_, kSignalHandlerEnabled);
      bool success;
      if (outlier_rejection) {
        map_optimization::OutlierRejectionSolverOptions outlier_rejection_options =
            map_optimization::OutlierRejectionSolverOptions::initFromFlags();
        success = optimizer.optimizeVisualInertial(
            options, missions_to_optimize, &outlier_rejection_options, 
            map.get(), representative_id, copy_next_representative_id,
            covariance, FLAGS_fix_rotation_all);
        // success = optimizer.optimizeVisualInertial(
        //     options, missions_to_optimize, &outlier_rejection_options, 
        //     map.get());
      } else {
        success = optimizer.optimizeVisualInertial(
            options, missions_to_optimize, nullptr, 
            map.get(), representative_id, copy_next_representative_id,
            covariance, FLAGS_fix_rotation_all);
        // success = optimizer.optimizeVisualInertial(
        //     options, missions_to_optimize, nullptr, 
        //     map.get());
        
      }

      if (!success) {
        return common::kUnknownError;
      }
      // compute mean of next representative w.r.t. current representative
      const pose::Transformation& pose1 = 
          (*(map.get())).getVertex(representative_id).get_T_M_I();
      const pose::Transformation& pose2 = 
          (*(map.get())).getVertex(copy_next_representative_id).get_T_M_I();
      
      pose::Transformation delta_pose = (pose2.inverse() * pose1).inverse();
      
      // residual:
      // [ position         ]   [ delta_p          ]
      // [ orientation (3x1)] = [ 2 * delta_q(0:2) ]
      Eigen::Matrix<double, 6, 6> covariance_;
      covariance_ << covariance[32], covariance[33], covariance[34],
                    2.0*covariance[28], 2.0*covariance[29], 2.0*covariance[30],
                    covariance[39], covariance[40], covariance[41],
                    2.0*covariance[35], 2.0*covariance[36], 2.0*covariance[37],
                    covariance[46], covariance[47], covariance[48],
                    2.0*covariance[42], 2.0*covariance[43], 2.0*covariance[44],
                    2.0*covariance[4], 2.0*covariance[5], 2.0*covariance[6],
                    4.0*covariance[0], 4.0*covariance[1], 4.0*covariance[2],
                    2.0*covariance[11], 2.0*covariance[12], 2.0*covariance[13],
                    4.0*covariance[7], 4.0*covariance[8], 4.0*covariance[9],
                    2.0*covariance[18], 2.0*covariance[19], 2.0*covariance[20],
                    4.0*covariance[14], 4.0*covariance[15], 4.0*covariance[16];
      
      Eigen::Matrix<double, 6, 6> information_ = covariance_.inverse();
      
      posegraph_error::Pose3d transformation_;
      transformation_.p = delta_pose.getPosition();
      transformation_.q = delta_pose.getRotation().toImplementation();
      
      vertexid2index =  vertex_id_to_index.find(representative_id);
      int index1 = vertexid2index->second;
      vertexid2index =  vertex_id_to_index.find(next_representative_id);
      int index2 = vertexid2index->second;

      posegraph_error::Constraint3d constraint;
      constraint.id_begin_ = index1;
      constraint.id_end_ = index2;
      constraint.transformation_ = transformation_;
      constraint.information_ = information_;
      constraints.push_back(constraint);
      // build map in the higher level
      // optimzie the map in the higher level
      // apply the rigid body transformation in the submap
      // fix representative and next_representative, perform optimization in each submap
  
    }
    kk++;
  }
  if (!FLAGS_optimize_posegraph){ 
    return common::kSuccess;
  }
  // --------optimization in the higher level map-------------- 
  ceres::Problem problem;
  
  CHECK(&map_of_poses != NULL);
  CHECK(&problem != NULL);
  if (constraints.empty()) {
    LOG(INFO) << "No constraints, no problem to optimize.";
    return common::kUnknownError;
  }

  ceres::LossFunction* loss_function = NULL;
  ceres::LocalParameterization* quaternion_local_parameterization =
      new ceres::EigenQuaternionParameterization;

  for (posegraph_error::VectorOfConstraints::const_iterator constraints_iter =
           constraints.begin();
       constraints_iter != constraints.end(); ++constraints_iter) {
    const posegraph_error::Constraint3d& constraint = *constraints_iter;

    posegraph_error::MapOfPoses::iterator pose_begin_iter = 
                      map_of_poses.find(constraint.id_begin_);

    posegraph_error::MapOfPoses::iterator pose_end_iter = 
                      map_of_poses.find(constraint.id_end_);

    const Eigen::Matrix<double, 6, 6> sqrt_information =
        constraint.information_.llt().matrixL();
    
    ceres::CostFunction* cost_function =
        posegraph_error::PoseGraphError::Create(constraint.transformation_, sqrt_information);

    problem.AddResidualBlock(cost_function, loss_function,
                              pose_begin_iter->second.p.data(),
                              pose_begin_iter->second.q.coeffs().data(),
                              pose_end_iter->second.p.data(),
                              pose_end_iter->second.q.coeffs().data());

    problem.SetParameterization(pose_begin_iter->second.q.coeffs().data(),
                                 quaternion_local_parameterization);
    problem.SetParameterization(pose_end_iter->second.q.coeffs().data(),
                                 quaternion_local_parameterization);
  }

  posegraph_error::MapOfPoses::iterator pose_start_iter = map_of_poses.begin();
  
  problem.SetParameterBlockConstant(pose_start_iter->second.p.data());
  problem.SetParameterBlockConstant(pose_start_iter->second.q.coeffs().data());
  //
  CHECK(&problem != NULL);

  ceres::Solver::Options options_;
  options_.max_num_iterations = 30;
  options_.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  ceres::Solver::Summary summary;
  ceres::Solve(options_, &problem, &summary);

  std::cout << summary.FullReport() << '\n';

  CHECK(summary.IsSolutionUsable())
      << "The solve was not successful, exiting.";
  if (!FLAGS_optimize_submaps_second){ 
    return common::kSuccess;
  }
  // ---------------optimization on the higher level is done---------
  // ---------------------apply rigid transformation-----------------
  posegraph_error::MapOfPoses::iterator map_of_pose_it;
  vi_map::MissionIdList missions_second_optimization;
  missions_second_optimization.clear();
  for (const pose_graph::VertexIdList& partition : partitioning){
    const pose_graph::VertexId& vertex_id = partition.front();
    vi_map::Vertex& vertex = (map.get())->getVertex(vertex_id);
    const vi_map::MissionId mission_id = vertex.getMissionId();
    const vi_map::MissionBaseFrame& original_mission_base_frame =
          (map.get())->getMissionBaseFrameForMission(mission_id);
    const pose::Transformation& T_G_M = original_mission_base_frame.get_T_G_M();
    vertexid2index = vertex_id_to_index.find(vertex_id);
    int index = vertexid2index -> second;
    map_of_pose_it = map_of_poses.find(index);
    posegraph_error::Pose3d pose = map_of_pose_it->second;
    // old pose is in mission frame
    const pose::Transformation& T_M_oldI = vertex.get_T_M_I();
    // new pose is in global frame
    pose::Transformation T_G_newI;
    T_G_newI.getRotation().toImplementation() = pose.q;
    T_G_newI.getPosition() = pose.p;

    pose::Transformation T_newI_oldI = T_G_newI.inverse() * T_G_M * T_M_oldI;
    // if the change of distance > 0.05 m
    // the mission needs to be optimized again
    const Eigen::Vector3d& delta_position = T_newI_oldI.getPosition();
    if (delta_position.squaredNorm() > 0.05){
      missions_second_optimization.insert(missions_second_optimization.end(),
                                          mission_id);
    }

    for (const pose_graph::VertexId& id: partition){
      vi_map::Vertex& vertex_ = (map.get())->getVertex(id);
      const pose::Transformation& T_M_oldI2 = vertex_.get_T_M_I();
      vertex_.set_T_M_I(T_M_oldI2 * T_newI_oldI.inverse());
    }

  }
  VertexIdToVertexIdIterator it_;
  for (it_ = source_to_dest_vertex_id_map.begin(); 
        it_ != source_to_dest_vertex_id_map.end(); it_ ++){
    const pose_graph::VertexId original_ = it_->first;
    const pose_graph::VertexId copy_ = it_->second;
    vi_map::Vertex& original_vertex_ = (map.get())->getVertex(original_);
    vi_map::Vertex& copy_vertex_ = (map.get())->getVertex(copy_);

    const vi_map::MissionId mission_id = original_vertex_.getMissionId();
    const vi_map::MissionBaseFrame& original_mission_base_frame =
          (map.get())->getMissionBaseFrameForMission(mission_id);
    const pose::Transformation& T_G_M1 = original_mission_base_frame.get_T_G_M();

    const vi_map::MissionId mission_id_2 = copy_vertex_.getMissionId();
    const vi_map::MissionBaseFrame& original_mission_base_frame_2 =
          (map.get())->getMissionBaseFrameForMission(mission_id_2);
    const pose::Transformation& T_G_M2 = original_mission_base_frame_2.get_T_G_M();
    
    const pose::Transformation& T_M1_originalI = original_vertex_.get_T_M_I();
    copy_vertex_.set_T_M_I(T_G_M2.inverse() * T_G_M1 * T_M1_originalI);
  }
  
  // return common::kSuccess;
  // -------------------optimization: 2nd time---------------------------
  unsigned int size_missions = missions_second_optimization.size();
  
  for(const vi_map::MissionId& new_mission : missions_second_optimization){
    // need to fix the copy of next representative
    if(new_mission != new_missions.back()){
      pose_graph::VertexIdList vertex_ids;
      (map.get())->getAllVertexIdsInMissionAlongGraph(new_mission, &vertex_ids);
      pose_graph::VertexId copy_next_representative_id = vertex_ids.back();
      vi_map::MissionIdSet missions_to_optimize;
      missions_to_optimize.clear();

      missions_to_optimize.insert(new_mission);
      map_optimization::VIMapOptimizer optimizer(plotter_, kSignalHandlerEnabled);
      bool success;
      if (outlier_rejection) {
        map_optimization::OutlierRejectionSolverOptions outlier_rejection_options =
            map_optimization::OutlierRejectionSolverOptions::initFromFlags();
        success = optimizer.optimizeVisualInertial(
            options, missions_to_optimize, &outlier_rejection_options, 
            map.get(), copy_next_representative_id, new_mission);
      } else {
        success = optimizer.optimizeVisualInertial(
            options, missions_to_optimize, nullptr, 
            map.get(), copy_next_representative_id, new_mission);
      }

      if (!success) {
        return common::kUnknownError;
      }
    }
    else{
      vi_map::MissionIdSet missions_to_optimize;
      missions_to_optimize.clear();

      missions_to_optimize.insert(new_mission);
      map_optimization::VIMapOptimizer optimizer(plotter_, kSignalHandlerEnabled);
      bool success;
      if (outlier_rejection) {
        map_optimization::OutlierRejectionSolverOptions outlier_rejection_options =
            map_optimization::OutlierRejectionSolverOptions::initFromFlags();
        success = optimizer.optimizeVisualInertial(
            options, missions_to_optimize, &outlier_rejection_options, 
            map.get());
      } else {
        success = optimizer.optimizeVisualInertial(
            options, missions_to_optimize, nullptr, 
            map.get());
      }

      if (!success) {
        return common::kUnknownError;
      }      
    }
  }
  time_t end_time;
  end_time = std::time(NULL);
  std::cout << "running time: " << difftime(end_time,start_time) << std::endl;
  // ----------------------------------------------------------
  return common::kSuccess;
}


void OptimizerPlugin::partitionMapinTimeOrder(
  const vi_map::VIMap& map, 
  const unsigned int num_of_vertex_per_submap,
  std::vector<pose_graph::VertexIdList>* partitioning){
  pose_graph::VertexIdList vertices;
  map.getAllVertexIds(&vertices);
  // only one mission
  const unsigned int num_partitions = std::ceil(1.0*vertices.size()/num_of_vertex_per_submap);

  CHECK_NOTNULL(partitioning)->clear();
  partitioning->resize(num_partitions);
  vi_map::MissionIdList mission_ids;
  map.getAllMissionIds(&mission_ids);

  for (const vi_map::MissionId& mission_id : mission_ids) {
    unsigned int count = 1;
    unsigned int index = 0;
    CHECK(mission_id.isValid());
    const vi_map::VIMission& mission = map.getMission(mission_id);
    const pose_graph::VertexId& id = mission.getRootVertexId();
    pose_graph::VertexId next_vertex_id;
    pose_graph::VertexId current_vertex_id = id;
    ((*partitioning)[index]).push_back(current_vertex_id);

    while (map.getNextVertex(current_vertex_id, map.getGraphTraversalEdgeType(mission_id), &next_vertex_id)){
      ((*partitioning)[index]).push_back(next_vertex_id);
      current_vertex_id = next_vertex_id;
      count = count + 1;
      
      if (count == num_of_vertex_per_submap){
        index = index + 1;
        count = 0;
      }

    }
  }
}


int OptimizerPlugin::relaxMap() {
  std::string selected_map_key;
  if (!getSelectedMapKeyIfSet(&selected_map_key)) {
    return common::kStupidUserError;
  }

  map_optimization::VIMapRelaxation relaxation(plotter_, kSignalHandlerEnabled);
  vi_map::VIMapManager map_manager;
  vi_map::VIMapManager::MapWriteAccess map =
      map_manager.getMapWriteAccess(selected_map_key);

  vi_map::MissionIdList mission_id_list;
  map.get()->getAllMissionIds(&mission_id_list);

  const bool success = relaxation.relax(mission_id_list, map.get());

  if (!success) {
    return common::kUnknownError;
  }
  return common::kSuccess;
}

int OptimizerPlugin::relaxMapMissionsSeparately() {
  std::string selected_map_key;
  if (!getSelectedMapKeyIfSet(&selected_map_key)) {
    return common::kStupidUserError;
  }

  map_optimization::VIMapRelaxation relaxation(plotter_, kSignalHandlerEnabled);
  vi_map::VIMapManager map_manager;
  vi_map::VIMapManager::MapWriteAccess map =
      map_manager.getMapWriteAccess(selected_map_key);

  vi_map::MissionIdList mission_id_list;
  map.get()->getAllMissionIds(&mission_id_list);

  for (const vi_map::MissionId& mission_id : mission_id_list) {
    relaxation.relax({mission_id}, map.get());
  }

  return common::kSuccess;
}

}  // namespace hierarchical_bundle_djustment_plugin

MAPLAB_CREATE_CONSOLE_PLUGIN_WITH_PLOTTER(
    hierarchical_bundle_djustment_plugin::OptimizerPlugin);
