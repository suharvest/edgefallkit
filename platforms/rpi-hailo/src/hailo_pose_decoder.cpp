#include "hailo_pose_decoder.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace rpi_hailo {
namespace {
float dq(const RawTensor&t,size_t pos){
  float v=t.info.format.type==HAILO_FORMAT_TYPE_UINT16
    ? reinterpret_cast<const uint16_t*>(t.data)[pos] : t.data[pos];
  return (v-t.info.quant_info.qp_zp)*t.info.quant_info.qp_scale;
}
struct Scale{const RawTensor*b=nullptr,*s=nullptr,*k=nullptr;int stride=0;};
float boxiou(const Detection&a,const Detection&b){return iou(a.box,b.box);}
}
std::vector<Detection> decodeYoloV8Pose(const std::vector<RawTensor>& ts,float threshold,float nms){
  std::vector<Scale> scales;
  for(int side:{80,40,20}){Scale g;g.stride=640/side;
    for(auto&t:ts){if((int)t.info.shape.height!=side||(int)t.info.shape.width!=side)continue;
      if(t.info.shape.features==64)g.b=&t;else if(t.info.shape.features==1)g.s=&t;else if(t.info.shape.features==51)g.k=&t;}
    if(g.b&&g.s&&g.k)scales.push_back(g);
  }
  std::vector<Detection> out;
  for(auto&g:scales){int h=g.s->info.shape.height,w=g.s->info.shape.width;
    for(int y=0;y<h;++y)for(int x=0;x<w;++x){size_t cell=size_t(y)*w+x;
      float score=dq(*g.s,cell);if(score<threshold)continue;
      std::array<float,4> dist{};
      for(int q=0;q<4;++q){float mx=-1e30f,sum=0;
        for(int n=0;n<16;++n)mx=std::max(mx,dq(*g.b,cell*64+q*16+n));
        for(int n=0;n<16;++n)sum+=std::exp(dq(*g.b,cell*64+q*16+n)-mx);
        for(int n=0;n<16;++n)dist[q]+=n*std::exp(dq(*g.b,cell*64+q*16+n)-mx)/sum;
        dist[q]*=g.stride;
      }
      float cx=(x+.5f)*g.stride,cy=(y+.5f)*g.stride;
      float left=cx-dist[0],top=cy-dist[1],right=cx+dist[2],bottom=cy+dist[3];
      Detection d;d.box={(left+right)/1280.f,(top+bottom)/1280.f,(right-left)/640.f,(bottom-top)/640.f,score};
      d.keypoints.resize(17);
      for(int j=0;j<17;++j){const size_t o=cell*51+j*3;
        float kx=dq(*g.k,o),ky=dq(*g.k,o+1),ks=dq(*g.k,o+2);
        d.keypoints[j].x=(g.stride*(2*kx-.5f)+cx)/640.f;
        d.keypoints[j].y=(g.stride*(2*ky-.5f)+cy)/640.f;
        d.keypoints[j].confidence=1.f/(1.f+std::exp(-ks));
      }
      out.push_back(std::move(d));
    }
  }
  std::sort(out.begin(),out.end(),[](auto&a,auto&b){return a.box.score>b.box.score;});
  std::vector<Detection> keep;for(auto&d:out){bool suppress=false;for(auto&k:keep)if(boxiou(d,k)>=nms){suppress=true;break;}if(!suppress)keep.push_back(std::move(d));}
  return keep;
}
}
