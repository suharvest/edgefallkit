#include "tracker_hailo.h"
#include <algorithm>
#include <cmath>

namespace rpi_hailo {
namespace {
float mid(const Pose& p, jetson_fall::Joint a, jetson_fall::Joint b, bool y) {
  const bool va=p.visible(a), vb=p.visible(b);
  if (!va && !vb) return 0;
  const auto pa=p.at(a), pb=p.at(b);
  const float aa=y?pa.y:pa.x, bb=y?pb.y:pb.x;
  return va&&vb ? (aa+bb)*.5f : (va?aa:bb);
}
float distance(const Box&a,const Box&b){return std::hypot(a.x-b.x,a.y-b.y);}
}
float iou(const Box&a,const Box&b){
  const float l=std::max(a.x-a.w/2,b.x-b.w/2), r=std::min(a.x+a.w/2,b.x+b.w/2);
  const float t=std::max(a.y-a.h/2,b.y-b.h/2), d=std::min(a.y+a.h/2,b.y+b.h/2);
  const float inter=std::max(0.f,r-l)*std::max(0.f,d-t);
  return inter/std::max(1e-9f,a.w*a.h+b.w*b.h-inter);
}
FallObservation observationFrom(const Track&t,double ts){
  FallObservation o; o.timestamp_sec=ts; o.person_score=t.score;
  const bool hip=t.pose.visible(jetson_fall::Joint::LeftHip)||t.pose.visible(jetson_fall::Joint::RightHip);
  const bool shoulder=t.pose.visible(jetson_fall::Joint::LeftShoulder)||t.pose.visible(jetson_fall::Joint::RightShoulder);
  o.valid=!t.pose.empty() && t.score>0 && hip && shoulder;
  if(!o.valid) return o;
  const float hx=mid(t.pose,jetson_fall::Joint::LeftHip,jetson_fall::Joint::RightHip,false);
  const float hy=mid(t.pose,jetson_fall::Joint::LeftHip,jetson_fall::Joint::RightHip,true);
  const float sx=mid(t.pose,jetson_fall::Joint::LeftShoulder,jetson_fall::Joint::RightShoulder,false);
  const float sy=mid(t.pose,jetson_fall::Joint::LeftShoulder,jetson_fall::Joint::RightShoulder,true);
  o.hip_y=hy; o.torso_angle_deg=std::atan2(std::abs(sx-hx),std::abs(sy-hy))*180.f/3.14159265f;
  o.bbox_aspect_ratio=t.box.h>1e-6f?t.box.w/t.box.h:0; return o;
}
std::vector<Track*> Tracker::update(const std::vector<Detection>& ds,double ts){
  std::vector<int> assignment(ds.size(),-1); std::vector<bool> used(tracks_.size(),false);
  struct M{float s;size_t d,t;}; std::vector<M> ms;
  for(size_t d=0;d<ds.size();++d)for(size_t t=0;t<tracks_.size();++t){
    float ov=iou(ds[d].box,tracks_[t]->box), dist=distance(ds[d].box,tracks_[t]->box);
    if(ov>=.2f||dist<=.25f)ms.push_back({ov+1-std::min(1.f,dist),d,t});}
  std::sort(ms.begin(),ms.end(),[](auto&a,auto&b){return a.s>b.s;});
  for(auto&m:ms)if(assignment[m.d]<0&&!used[m.t]){assignment[m.d]=(int)m.t;used[m.t]=true;}
  for(size_t d=0;d<ds.size();++d){Track* t=nullptr;
    if(assignment[d]>=0)t=tracks_[assignment[d]].get(); else {auto n=std::make_unique<Track>();n->id=next_id_++;tracks_.push_back(std::move(n));t=tracks_.back().get();}
    t->box=ds[d].box;t->score=ds[d].box.score;t->pose=Pose(ds[d].keypoints,640,640,keypoint_threshold_);t->missed=0;++t->age;
    t->observation=observationFrom(*t,ts);
    auto p=t->temporal.update(jetson_fall::makeTemporalFrame(&t->pose,t->observation,640,640),ts);
    t->observation.temporal_available=t->observation.valid;
    t->observation.temporal_positive=t->observation.valid&&p.positive;t->observation.temporal_probability=p.probability;
    t->output=t->fall.update(t->observation);
  }
  const size_t old=used.size();for(size_t t=0;t<old;++t)if(!used[t]){
    auto&x=*tracks_[t];++x.missed;FallObservation o;o.timestamp_sec=ts;
    // Missing/invalid observations may retain/expire state, never originate an event.
    o.temporal_available=false;x.observation=o;x.output=x.fall.update(o);
  }
  tracks_.erase(std::remove_if(tracks_.begin(),tracks_.end(),[](auto&p){return p->missed>8;}),tracks_.end());
  std::vector<Track*> out;for(auto&p:tracks_)if(!p->missed)out.push_back(p.get());return out;
}
}
