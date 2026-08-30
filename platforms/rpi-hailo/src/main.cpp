#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/hailo/tensor_meta.hpp>
#include <glib-unix.h>
#ifdef HAVE_MOSQUITTO
#include <mosquitto.h>
#endif
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "batch_policy.h"
#include "batched_hailo_runner.h"
#include "frame_batcher.h"
#include "frame_processing_metrics.h"
#include "hailo_pose_decoder.h"
#include "runner_lifecycle.h"
#include "runtime_config.h"

using Clock=std::chrono::steady_clock;
namespace {
std::string env(const char*n,const char*d=""){auto*v=std::getenv(n);return v?v:d;}
double timeSeconds(Clock::time_point t){return std::chrono::duration<double>(t.time_since_epoch()).count();}
double now(){return timeSeconds(Clock::now());}
int64_t nowNs(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
Clock::time_point fromNs(int64_t n){return Clock::time_point(std::chrono::nanoseconds(n));}
uint64_t epochMs(){return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();}
std::string replace(std::string s,const std::string&a,const std::string&b){size_t p;while((p=s.find(a))!=std::string::npos)s.replace(p,a.size(),b);return s;}
std::string jsonEscape(const std::string& value){std::ostringstream o;for(unsigned char c:value){switch(c){case '"':o<<"\\\"";break;case '\\':o<<"\\\\";break;case '\b':o<<"\\b";break;case '\f':o<<"\\f";break;case '\n':o<<"\\n";break;case '\r':o<<"\\r";break;case '\t':o<<"\\t";break;default:if(c<0x20)o<<"\\u00"<<std::hex<<std::setw(2)<<std::setfill('0')<<(int)c<<std::dec<<std::setfill(' ');else o<<c;}}return o.str();}
struct App;
struct Stream {std::string id,url;GstElement*pipeline=nullptr,*appsink=nullptr;gulong sample_handler=0;guint bus_watch=0;int index=0;uint64_t next_seq=0;rpi_hailo::Tracker tracker;App*app=nullptr;uint64_t frames=0,bench_frames=0,global_event_id=0;double bench_latency_sum=0;std::mutex metrics_mutex;std::atomic<int64_t> input_at_ns{0};};
struct App {
#ifdef HAVE_MOSQUITTO
mosquitto*mqtt=nullptr;
#endif
std::string topic;float score=.35f,kpt=.25f;int benchmark_seconds=0,warmup_seconds=0;double benchmark_started=0;std::atomic<bool> benchmark_active{false};rpi_hailo::RunnerLifecycle runner_lifecycle;std::vector<std::unique_ptr<Stream>> streams;GMainLoop*loop=nullptr;std::mutex benchmark_mutex;std::unique_ptr<rpi_hailo::FrameBatcher> batcher;std::unique_ptr<rpi_hailo::BatchedHailoRunner> runner;};

std::string payload(Stream&s,const std::vector<rpi_hailo::Track*>& ps,rpi_hailo::ProcessingBackend backend,float pipeline_ms,float decode_ms=0.f,float track_ms=0.f,float pipeline_full_ms=0.f){
  int fallen=0,event_edges=0;const rpi_hailo::Track*primary=nullptr;const char*state="normal";
  for(auto&p:s.tracker.tracks()){fallen+=p->output.fall_detected;event_edges+=p->output.fall_event?1:0;if(p->output.state==jetson_fall::FallState::Fallen)state="fallen";else if(std::string(state)=="normal"&&p->output.state==jetson_fall::FallState::Recovering)state="recovering";else if(std::string(state)=="normal"&&p->output.state==jetson_fall::FallState::Suspected)state="suspected";}
  s.global_event_id+=event_edges;for(auto*p:ps)if(!primary||p->score>primary->score)primary=p;uint64_t frame_id;{std::lock_guard<std::mutex> lock(s.metrics_mutex);frame_id=s.frames;}
  std::ostringstream o;o<<std::fixed<<std::setprecision(4)<<"{\"timestamp\":"<<epochMs()<<",\"frame_id\":"<<frame_id
    <<",\"pipeline_ms\":"<<pipeline_ms<<",\"pipeline_time_ms\":"<<pipeline_ms<<",\"decode_ms\":"<<decode_ms<<",\"track_ms\":"<<track_ms<<",\"pipeline_full_ms\":"<<pipeline_full_ms
    <<",\"pipeline_full_metric\":\""<<rpi_hailo::pipelineFullMetric(backend)<<"\",\"inference_time_ms\":0.0,\"inference_time_metric\":\"unavailable\",\"latency_metric\":\""<<rpi_hailo::pipelineMetric(backend)<<"\",\"stream_id\":\""<<jsonEscape(s.id)<<"\",\"fall_detected\":"<<(fallen?"true":"false")
    <<",\"fall_event\":"<<(event_edges?"true":"false")<<",\"event_id\":"<<s.global_event_id<<",\"event_id_scope\":\"stream_global_event_id\",\"global_event_id\":"<<s.global_event_id<<",\"state\":\""<<state<<"\",\"person_detected\":"<<(!ps.empty()?"true":"false")
    <<",\"person_count\":"<<ps.size()<<",\"fallen_count\":"<<fallen<<",\"tracking\":"<<(!s.tracker.tracks().empty()?"true":"false")<<",\"persons\":[";
  bool first=true;for(auto&owned:s.tracker.tracks()){auto*p=owned.get();if(!first)o<<',';first=false;o<<"{\"track_id\":"<<p->id<<",\"person_detected\":"<<(p->missed==0?"true":"false")<<",\"person_score\":"<<p->score
    <<",\"fall_detected\":"<<(p->output.fall_detected?"true":"false")<<",\"fall_event\":"<<(p->output.fall_event?"true":"false")
    <<",\"event_id\":"<<p->output.event_id<<",\"state\":\""<<jetson_fall::fallStateName(p->output.state)<<"\",\"tracking\":"<<(p->missed==0?"true":"false")<<",\"missed_frames\":"<<p->missed<<",\"bbox\":["
    <<p->box.x<<','<<p->box.y<<','<<p->box.w<<','<<p->box.h<<"],\"features\":{\"valid\":"<<(p->observation.valid?"true":"false")<<",\"person_score\":"<<p->score<<",\"hip_y\":"<<p->observation.hip_y<<",\"hip_drop_speed\":"<<p->output.diagnostics.hip_drop_speed<<",\"hip_drop_distance\":"<<p->output.diagnostics.hip_drop_distance<<",\"torso_angle_deg\":"<<p->observation.torso_angle_deg<<",\"bbox_aspect_ratio\":"<<p->observation.bbox_aspect_ratio<<",\"temporal_probability\":"<<p->observation.temporal_probability<<",\"temporal_positive\":"<<(p->observation.temporal_positive?"true":"false")<<"},\"pose17\":[";
    if(p->missed==0)for(int j=0;j<17;++j){if(j)o<<',';auto joint=static_cast<jetson_fall::Joint>(j);auto point=p->pose.at(joint);o<<'['<<point.x/640.f<<','<<point.y/640.f<<','<<p->pose.confidence(joint)<<']';}o<<"],\"keypoints\":[]}";}o<<']';
  if(primary)o<<",\"features\":{\"hip_y\":"<<primary->observation.hip_y<<",\"person_score\":"<<primary->score<<",\"hip_drop_speed\":"<<primary->output.diagnostics.hip_drop_speed<<",\"hip_drop_distance\":"<<primary->output.diagnostics.hip_drop_distance<<",\"torso_angle_deg\":"<<primary->observation.torso_angle_deg<<",\"bbox_aspect_ratio\":"<<primary->observation.bbox_aspect_ratio<<",\"temporal_probability\":"<<primary->observation.temporal_probability<<",\"temporal_positive\":"<<(primary->observation.temporal_positive?"true":"false")<<",\"valid\":"<<(primary->observation.valid?"true":"false")<<"},\"keypoints\":[],\"pose17\":[";
  if(primary)for(int j=0;j<17;++j){if(j)o<<',';auto joint=static_cast<jetson_fall::Joint>(j);auto point=primary->pose.at(joint);o<<'['<<point.x/640.f<<','<<point.y/640.f<<','<<primary->pose.confidence(joint)<<']';}if(primary)o<<']';
  if(!primary)o<<",\"features\":{\"valid\":false,\"person_score\":0.0,\"hip_y\":0.0,\"hip_drop_speed\":0.0,\"hip_drop_distance\":0.0,\"torso_angle_deg\":0.0,\"bbox_aspect_ratio\":0.0,\"temporal_probability\":0.0,\"temporal_positive\":false},\"keypoints\":[],\"pose17\":[]";o<<'}';return o.str();
}

void processFrame(Stream&s,const std::vector<rpi_hailo::RawTensor>&tensors,rpi_hailo::ProcessingBackend backend,Clock::time_point latency_origin,Clock::time_point output_timestamp){
  const auto started=Clock::now();auto ds=rpi_hailo::decodeYoloV8Pose(tensors,s.app->score,.7f);const auto decoded=Clock::now();auto persons=s.tracker.update(ds,rpi_hailo::trackerTimestampSeconds(backend,timeSeconds(output_timestamp),timeSeconds(latency_origin)));const auto tracked=Clock::now();float latency=std::chrono::duration<float,std::milli>(started-latency_origin).count();
  bool print_sample=false;{std::lock_guard<std::mutex> lock(s.metrics_mutex);++s.frames;print_sample=s.frames<=2;if(s.app->benchmark_active){++s.bench_frames;s.bench_latency_sum+=latency;}}
  float decode_ms=std::chrono::duration<float,std::milli>(decoded-started).count();float track_ms=std::chrono::duration<float,std::milli>(tracked-decoded).count();float pipeline_full_ms=std::chrono::duration<float,std::milli>(tracked-latency_origin).count();auto json=payload(s,persons,backend,latency,decode_ms,track_ms,pipeline_full_ms);auto topic=replace(s.app->topic,"{stream_id}",s.id);
#ifdef HAVE_MOSQUITTO
  if(s.app->mqtt)mosquitto_publish(s.app->mqtt,nullptr,topic.c_str(),json.size(),json.data(),0,false);
#else
  (void)topic;if(print_sample)std::cout<<json<<"\n";
#endif
}
GstPadProbeReturn preProbe(GstPad*,GstPadProbeInfo*,gpointer data){static_cast<Stream*>(data)->input_at_ns=nowNs();return GST_PAD_PROBE_OK;}
GstPadProbeReturn outProbe(GstPad*,GstPadProbeInfo*info,gpointer data){auto&s=*static_cast<Stream*>(data);GstBuffer*buf=GST_PAD_PROBE_INFO_BUFFER(info);if(!buf)return GST_PAD_PROBE_OK;const auto output_timestamp=Clock::now();std::vector<rpi_hailo::RawTensor> tensors;struct Mapping{GstBuffer*b;GstMapInfo m;};std::vector<Mapping> maps;gpointer state=nullptr;GstMeta*meta=nullptr;while((meta=gst_buffer_iterate_meta_filtered(buf,&state,GST_PARENT_BUFFER_META_API_TYPE))){auto*p=reinterpret_cast<GstParentBufferMeta*>(meta);GstMapInfo mi{};if(!gst_buffer_map(p->buffer,&mi,GST_MAP_READ))continue;auto*tm=GST_TENSOR_META_GET(p->buffer);if(tm){maps.push_back({p->buffer,mi});tensors.push_back({mi.data,tm->info});}else gst_buffer_unmap(p->buffer,&mi);}processFrame(s,tensors,rpi_hailo::ProcessingBackend::Legacy,fromNs(s.input_at_ns.load()),output_timestamp);for(auto&m:maps)gst_buffer_unmap(m.b,&m.m);return GST_PAD_PROBE_OK;}
GstFlowReturn newSample(GstAppSink*sink,gpointer data){auto&s=*static_cast<Stream*>(data);GstSample*sample=gst_app_sink_pull_sample(sink);if(!sample)return GST_FLOW_EOS;GstBuffer*buffer=gst_sample_get_buffer(sample);GstMapInfo map{};if(!buffer||!gst_buffer_map(buffer,&map,GST_MAP_READ)){gst_sample_unref(sample);return GST_FLOW_ERROR;}constexpr size_t bytes=640U*640U*3U;GstFlowReturn result=GST_FLOW_OK;if(map.size!=bytes){std::cerr<<"stream "<<s.id<<" produced invalid RGB frame size "<<map.size<<"\n";result=GST_FLOW_ERROR;}else{rpi_hailo::BatchFrame frame;frame.stream=s.index;frame.seq=s.next_seq++;frame.timestamp=Clock::now();frame.rgb.assign(map.data,map.data+bytes);s.app->batcher->enqueue(std::move(frame));}gst_buffer_unmap(buffer,&map);gst_sample_unref(sample);return result;}
gboolean busCb(GstBus*,GstMessage*m,gpointer data){auto*s=static_cast<Stream*>(data);if(GST_MESSAGE_TYPE(m)==GST_MESSAGE_ERROR){GError*e=nullptr;gchar*d=nullptr;gst_message_parse_error(m,&e,&d);std::cerr<<"pipeline "<<s->id<<": "<<e->message<<"\n";g_error_free(e);g_free(d);g_main_loop_quit(s->app->loop);}else if(GST_MESSAGE_TYPE(m)==GST_MESSAGE_EOS)g_main_loop_quit(s->app->loop);return TRUE;}
gboolean quitLoop(gpointer data){g_main_loop_quit(static_cast<App*>(data)->loop);return G_SOURCE_REMOVE;}
gboolean quitPostedLoop(gpointer data){g_main_loop_quit(static_cast<GMainLoop*>(data));return G_SOURCE_REMOVE;}
void unrefPostedLoop(gpointer data){g_main_loop_unref(static_cast<GMainLoop*>(data));}
void postLoopQuit(GMainLoop*loop){if(loop)g_main_context_invoke_full(nullptr,G_PRIORITY_DEFAULT,quitPostedLoop,g_main_loop_ref(loop),unrefPostedLoop);}
gboolean finish(gpointer data){auto*a=static_cast<App*>(data);std::lock_guard<std::mutex> app_lock(a->benchmark_mutex);a->benchmark_active=false;double sec=now()-a->benchmark_started;for(auto&sp:a->streams){std::lock_guard<std::mutex> lock(sp->metrics_mutex);std::cout<<"BENCHMARK stream="<<sp->id<<" frames="<<sp->bench_frames<<" seconds="<<sec<<" fps="<<(sp->bench_frames/sec)<<" mean_pipeline_ms="<<(sp->bench_frames?sp->bench_latency_sum/sp->bench_frames:0)<<" warmup_seconds="<<a->warmup_seconds<<"\n";}g_main_loop_quit(a->loop);return G_SOURCE_REMOVE;}
gboolean startBenchmark(gpointer data){auto*a=static_cast<App*>(data);std::lock_guard<std::mutex> app_lock(a->benchmark_mutex);for(auto&sp:a->streams){std::lock_guard<std::mutex> lock(sp->metrics_mutex);sp->bench_frames=0;sp->bench_latency_sum=0;}a->benchmark_started=now();a->benchmark_active=true;g_timeout_add_seconds(a->benchmark_seconds,finish,a);return G_SOURCE_REMOVE;}
std::string pipelinePrefix(const Stream&s,int latency,bool drop){std::ostringstream q;if(s.url.rfind("test://",0)==0)q<<"videotestsrc is-live=true pattern=ball ! video/x-raw,framerate=30/1 ";else if(s.url.rfind("file://",0)==0)q<<"uridecodebin uri=\""<<s.url<<"\" ! videorate ! video/x-raw,framerate=15/1 ";else q<<"rtspsrc location=\""<<s.url<<"\" latency="<<latency<<" protocols=tcp"<<(drop?" drop-on-latency=true":"")<<" ! rtph264depay ! h264parse ! decodebin ";q<<"! videoconvert ! videoscale ! video/x-raw,format=RGB,width=640,height=640 ";return q.str();}
void attachBus(Stream&s){auto*bus=gst_element_get_bus(s.pipeline);s.bus_watch=gst_bus_add_watch(bus,busCb,&s);gst_object_unref(bus);}
bool legacyPipeline(Stream&s,const std::string&hef,int latency,bool drop,int depth){std::ostringstream q;q<<pipelinePrefix(s,latency,drop)<<"! queue max-size-buffers="<<depth<<" max-size-bytes=0 max-size-time=0 leaky=downstream ! identity name=pre ! hailonet name=net hef-path=\""<<hef<<"\" scheduling-algorithm=1 vdevice-group-id=fall-shared ! fakesink sync=false";GError*e=nullptr;s.pipeline=gst_parse_launch(q.str().c_str(),&e);if(!s.pipeline){std::cerr<<(e?e->message:"pipeline creation failed")<<"\n";if(e)g_error_free(e);return false;}auto*pre=gst_bin_get_by_name(GST_BIN(s.pipeline),"pre");auto*net=gst_bin_get_by_name(GST_BIN(s.pipeline),"net");auto*pp=gst_element_get_static_pad(pre,"src");auto*np=gst_element_get_static_pad(net,"src");gst_pad_add_probe(pp,GST_PAD_PROBE_TYPE_BUFFER,preProbe,&s,nullptr);gst_pad_add_probe(np,GST_PAD_PROBE_TYPE_BUFFER,outProbe,&s,nullptr);gst_object_unref(pp);gst_object_unref(np);gst_object_unref(pre);gst_object_unref(net);attachBus(s);return true;}
bool sharedPipeline(Stream&s,int latency,bool drop){std::string q=pipelinePrefix(s,latency,drop)+"! appsink name=sink max-buffers=2 drop=true sync=false emit-signals=true";GError*e=nullptr;s.pipeline=gst_parse_launch(q.c_str(),&e);if(!s.pipeline){std::cerr<<(e?e->message:"pipeline creation failed")<<"\n";if(e)g_error_free(e);return false;}s.appsink=gst_bin_get_by_name(GST_BIN(s.pipeline),"sink");s.sample_handler=g_signal_connect(s.appsink,"new-sample",G_CALLBACK(newSample),&s);attachBus(s);return true;}
void stopAll(App&a){for(auto&s:a.streams)if(s->appsink&&s->sample_handler){g_signal_handler_disconnect(s->appsink,s->sample_handler);s->sample_handler=0;}for(auto&s:a.streams)if(s->pipeline){gst_element_set_state(s->pipeline,GST_STATE_NULL);gst_element_get_state(s->pipeline,nullptr,nullptr,5*GST_SECOND);}if(a.runner)a.runner->stop();else if(a.batcher)a.batcher->stop(true);for(auto&s:a.streams){if(s->bus_watch){g_source_remove(s->bus_watch);s->bus_watch=0;}if(s->appsink)gst_object_unref(s->appsink);if(s->pipeline)gst_object_unref(s->pipeline);s->appsink=nullptr;s->pipeline=nullptr;}}
}

int main(int argc,char**argv){gst_init(&argc,&argv);
#ifdef HAVE_MOSQUITTO
mosquitto_lib_init();
#endif
App app;int exit_code=0;try{app.topic=env("MQTT_TOPIC","recamera/fall-detection/results/{stream_id}");app.score=std::stof(env("SCORE_THRESHOLD","0.35"));app.kpt=std::stof(env("KEYPOINT_THRESHOLD","0.25"));
#ifdef HAVE_MOSQUITTO
  app.mqtt=mosquitto_new(nullptr,true,nullptr);auto user=env("MQTT_USERNAME"),pass=env("MQTT_PASSWORD");if(!user.empty())mosquitto_username_pw_set(app.mqtt,user.c_str(),pass.c_str());if(mosquitto_connect(app.mqtt,env("MQTT_HOST","127.0.0.1").c_str(),std::stoi(env("MQTT_PORT","1883")),30)==MOSQ_ERR_SUCCESS)mosquitto_loop_start(app.mqtt);else{mosquitto_destroy(app.mqtt);app.mqtt=nullptr;}
#endif
  std::string list=env("STREAMS");size_t start=0;while(start<list.size()){size_t end=list.find(';',start);auto item=list.substr(start,end-start);auto sep=item.find('|');if(sep!=std::string::npos){auto s=std::make_unique<Stream>();s->id=item.substr(0,sep);s->url=item.substr(sep+1);s->index=static_cast<int>(app.streams.size());s->app=&app;s->tracker=rpi_hailo::Tracker(app.kpt);app.streams.push_back(std::move(s));}if(end==std::string::npos)break;start=end+1;}if(app.streams.empty()){std::cerr<<"STREAMS must contain id|rtsp-url\n";return 2;}
  const std::string hef=env("HEF_PATH","/models/yolov8s_pose.hef");const int latency=std::stoi(env("RTSP_LATENCY_MS","100"));int queue_depth;bool drop;auto config=rpi_hailo::parseBatchMode("auto");int wait_ms=20;try{queue_depth=rpi_hailo_config::parseQueueDepth(env("INFERENCE_QUEUE_DEPTH","1"));drop=rpi_hailo_config::parseDropOnLatency(env("RTSP_DROP_ON_LATENCY","false"));app.warmup_seconds=rpi_hailo_config::parseNonNegativeInt("BENCHMARK_WARMUP_SECONDS",env("BENCHMARK_WARMUP_SECONDS","0"),600);app.benchmark_seconds=rpi_hailo_config::parseNonNegativeInt("BENCHMARK_SECONDS",env("BENCHMARK_SECONDS","0"));config=rpi_hailo::parseBatchMode(env("HAILO_BATCH_MODE","auto"));wait_ms=rpi_hailo::parseBatchWaitMs(env("HAILO_BATCH_WAIT_MS","20"));}catch(const std::exception&e){std::cerr<<"Invalid environment: "<<e.what()<<"\n";return 2;}
  auto context=rpi_hailo::BatchedHailoRunner::inspectHef(hef);auto decision=rpi_hailo::chooseBatch(config,context.network_groups,context.multi_context,static_cast<int>(app.streams.size()));std::cout<<"HAILO_BATCH mode="<<rpi_hailo::batchModeName(config)<<" backend="<<(decision.shared?"shared":"legacy")<<" batch="<<decision.batch_size<<" streams="<<app.streams.size()<<" network_groups="<<context.network_groups<<" multi_context="<<(context.multi_context?1:0)<<std::endl;
  app.loop=g_main_loop_new(nullptr,FALSE);if(decision.shared){app.batcher=std::make_unique<rpi_hailo::FrameBatcher>(app.streams.size(),decision.batch_size,wait_ms,2);app.runner=std::make_unique<rpi_hailo::BatchedHailoRunner>(hef,decision.batch_size,*app.batcher,[&app](rpi_hailo::BatchFrame&&frame,const std::vector<rpi_hailo::RawTensor>&tensors){if(frame.stream<0||static_cast<size_t>(frame.stream)>=app.streams.size())throw std::runtime_error("invalid result stream");processFrame(*app.streams[frame.stream],tensors,rpi_hailo::ProcessingBackend::Shared,frame.timestamp,Clock::now());},[&app](const std::string&e){std::cerr<<"shared Hailo runner: "<<e<<"\n";app.runner_lifecycle.fail([loop=app.loop]{postLoopQuit(loop);});});app.runner->start();}
  for(auto&s:app.streams){bool ok=decision.shared?sharedPipeline(*s,latency,drop):legacyPipeline(*s,hef,latency,drop,queue_depth);if(!ok)throw std::runtime_error("pipeline creation failed");if(gst_element_set_state(s->pipeline,GST_STATE_PLAYING)==GST_STATE_CHANGE_FAILURE)throw std::runtime_error("pipeline failed to start");}
  guint sigint=g_unix_signal_add(SIGINT,quitLoop,&app),sigterm=g_unix_signal_add(SIGTERM,quitLoop,&app);if(app.benchmark_seconds>0){if(app.warmup_seconds>0)g_timeout_add_seconds(app.warmup_seconds,startBenchmark,&app);else startBenchmark(&app);}if(app.runner_lifecycle.shouldRunLoop())g_main_loop_run(app.loop);if(sigint&&g_main_context_find_source_by_id(nullptr,sigint))g_source_remove(sigint);if(sigterm&&g_main_context_find_source_by_id(nullptr,sigterm))g_source_remove(sigterm);exit_code=app.runner_lifecycle.exitCode(exit_code);
}catch(const std::exception&e){std::cerr<<"fall-hailo: "<<e.what()<<"\n";exit_code=3;}stopAll(app);exit_code=app.runner_lifecycle.exitCode(exit_code);
#ifdef HAVE_MOSQUITTO
if(app.mqtt){mosquitto_loop_stop(app.mqtt,true);mosquitto_disconnect(app.mqtt);mosquitto_destroy(app.mqtt);}mosquitto_lib_cleanup();
#endif
if(app.loop)g_main_loop_unref(app.loop);return exit_code;}
