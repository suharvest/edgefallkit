#include "tracker_hailo.h"
#include <cassert>
#include <cmath>
#include <iostream>
int main(){
  rpi_hailo::Box a{.5f,.5f,.2f,.4f,.9f},b{.5f,.5f,.2f,.4f,.8f};
  assert(rpi_hailo::iou(a,b)>.99f);
  rpi_hailo::Tracker tracker;
  auto active=tracker.update({},0.0); assert(active.empty());
  rpi_hailo::Track incomplete; incomplete.score=.9f;
  std::vector<jetson_fall::Keypoint> points(17); points[0]={.5f,.5f,.9f};
  incomplete.pose=jetson_fall::Pose(points,640,640,.25f);
  assert(!rpi_hailo::observationFrom(incomplete,1.0).valid);
  // hip_y is normalised by the 640 px model frame; one-side anchors suffice.
  rpi_hailo::Track one_side; one_side.score=.9f; one_side.box={.5f,.5f,.2f,.4f,.9f};
  std::vector<jetson_fall::Keypoint> torso(17);
  torso[5]={.5f,.3f,.9f}; torso[12]={.5f,.6f,.9f};
  one_side.pose=jetson_fall::Pose(torso,640,640,.25f);
  const auto o=rpi_hailo::observationFrom(one_side,1.0);
  assert(o.valid && std::abs(o.hip_y-.6f)<1e-4f);
  one_side.box.h=0; assert(!rpi_hailo::observationFrom(one_side,1.0).valid);
  std::cout<<"hailo logic test passed\n";
}
