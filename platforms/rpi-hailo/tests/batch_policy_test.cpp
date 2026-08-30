#include "batch_policy.h"
#include "frame_batcher.h"
#include <chrono>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace rpi_hailo;
static void ok(bool x, const char *m) { if (!x) throw std::runtime_error(m); }
static void bad(const std::function<void()> &f) { bool x=false; try { f(); } catch (const std::invalid_argument &) { x=true; } ok(x,"expected invalid"); }
int main() {
  ok(parseBatchMode("auto").mode == BatchMode::Auto,"auto"); ok(parseBatchMode("4").fixed_size == 4,"fixed");
  for (const auto &v : {"AUTO","2","9",""}) bad([&]{ parseBatchMode(v); });
  ok(parseBatchWaitMs("0") == 0 && parseBatchWaitMs("1000") == 1000,"wait"); bad([]{parseBatchWaitMs("1001");});
  ok(!chooseBatch(parseBatchMode("auto"),1,false,8).shared,"single legacy");
  ok(chooseBatch(parseBatchMode("auto"),1,true,4).batch_size == 4,"auto four");
  ok(chooseBatch(parseBatchMode("auto"),2,true,4).shared == false,"multiple groups");
  ok(chooseBatch(parseBatchMode("off"),1,true,8).shared == false,"off");
  ok(chooseBatch(parseBatchMode("1"),1,false,1).shared,"fixed override");
  FrameBatcher b(3,4,0,2); auto t=std::chrono::steady_clock::now();
  for (int i=0;i<3;++i) { b.enqueue({0,uint64_t(i),t,{}}); }
  auto s=b.stats(); ok(s.drops[0]==1,"fifo drop"); std::vector<BatchFrame> out; ok(b.take(out),"take"); ok(out.size()==2 && out[0].seq==1 && out[1].seq==2,"oldest retained");
  b.enqueue({0,3,t,{}}); b.enqueue({1,9,t,{}}); b.enqueue({2,8,t,{}}); b.enqueue({1,10,t,{}}); ok(b.take(out),"fair"); ok(out.size()==4 && out[0].stream==1,"rr first");
  b.stop(); ok(!b.enqueue({0,99,t,{}}),"stop enqueue"); ok(!b.take(out),"stop take");
  FrameBatcher timed(2,4,20,2); auto future=std::chrono::steady_clock::now()+std::chrono::milliseconds(12); timed.enqueue({0,1,future,{}});
  auto begin=std::chrono::steady_clock::now(); ok(timed.take(out),"partial timeout"); auto elapsed=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-begin).count(); ok(elapsed>=10 && elapsed<100,"deadline wait"); timed.stop();
  FrameBatcher full(4,4,200,2); for(int i=0;i<4;++i) full.enqueue({i,uint64_t(i),std::chrono::steady_clock::now(),{}});
  begin=std::chrono::steady_clock::now(); ok(full.take(out)&&out.size()==4,"full batch"); elapsed=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-begin).count(); ok(elapsed<50,"full batch immediate"); full.stop();
  std::cout << "batch policy/batcher test passed\n";
}
