#include "tracker_hailo.h"
#include <cassert>
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
  std::cout<<"hailo logic test passed\n";
}
