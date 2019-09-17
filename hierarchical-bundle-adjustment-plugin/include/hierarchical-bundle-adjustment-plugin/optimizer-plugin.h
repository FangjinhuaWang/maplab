#ifndef HIERARCHICAL_BUNDLE_ADJUSTMENT_OPTIMIZER_PLUGIN_H_
#define HIERARCHICAL_BUNDLE_ADJUSTMENT_OPTIMIZER_PLUGIN_H_

#include <string>
#include <math.h>
#include <ceres/ceres.h>
#include <console-common/basic-console-plugin.h>
#include <console-common/console.h>
#include <map-manager/map-manager.h>
#include <map-optimization/outlier-rejection-solver.h>
#include <map-optimization/vi-optimization-builder.h>
#include <vi-map/vi-map.h>
#include <visualization/viwls-graph-plotter.h>
// first use metics to partition the graph
#include <vi-map-helpers/vi-map-partitioner.h>

#include "map-optimization/vi-map-optimizer.h"
#include "map-optimization/vi-map-relaxation.h"
#include <console-common/console-plugin-base-with-plotter.h>

#include "hierarchical-bundle-adjustment-plugin/posegraph_error.h"

namespace common {
class Console;
}  // namespace common

namespace visualization {
class ViwlsGraphRvizPlotter;
}  // namespace visualization

namespace hierarchical_bundle_djustment_plugin {
class OptimizerPlugin : public common::ConsolePluginBaseWithPlotter {
 public:
  OptimizerPlugin(
      common::Console* console, visualization::ViwlsGraphRvizPlotter* plotter);

  virtual std::string getPluginId() const {
    return "hierarchical-bundle-djustment-plugin";
  }

 private:
  int optimizeVisualInertial_naive(bool visual_only, bool outlier_rejection);

  int relaxMap();
  int relaxMapMissionsSeparately();
  
  void partitionMapinTimeOrder(
  const vi_map::VIMap& map, 
  const unsigned int num_of_vertex_per_submap,
  std::vector<pose_graph::VertexIdList>* partitioning);




  static constexpr bool kSignalHandlerEnabled = true;
};
}  // namespace hierarchical_bundle_djustment_plugin
#endif  // HIERARCHICAL_BUNDLE_ADJUSTMENT_OPTIMIZER_PLUGIN_H_
