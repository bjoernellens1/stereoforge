// Synthetic, separately paced transport experiment; not a camera or GPU benchmark.
#include "payload.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <time.h>
#include <vector>
using clock_type=std::chrono::steady_clock;
using ns=std::chrono::nanoseconds;
static int64_t wall(){return std::chrono::duration_cast<ns>(clock_type::now().time_since_epoch()).count();}
static int64_t cpu(){timespec t{}; if(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t))throw std::runtime_error("thread CPU clock failed");return int64_t(t.tv_sec)*1000000000+t.tv_nsec;}
struct Pub {int64_t id,target,begin,dwall,dcpu,rwall,rcpu;};
struct Rec {int64_t id,at; char kind;};
int main(int argc,char**argv){
 try{
  if(argc!=5)throw std::invalid_argument("usage: transport pub|sub output.csv measured_seconds warmup_seconds");
  std::string role=argv[1],path=argv[2];int duration=std::stoi(argv[3]),warmup=std::stoi(argv[4]);
  if((role!="pub"&&role!="sub")||duration<1||duration>600||warmup<0||warmup>30)throw std::invalid_argument("invalid options");
  rclcpp::init(0,nullptr); auto node=std::make_shared<rclcpp::Node>("stereoforge_transport_"+role,rclcpp::NodeOptions().use_intra_process_comms(false).start_parameter_services(false).start_parameter_event_publisher(false));
  auto qos=rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
  std::cout<<"rmw="<<rmw_get_implementation_identifier()<<"\n";
  constexpr int64_t period=1000000000LL; const int count=duration*60;
  if(role=="pub"){
   auto dp=node->create_publisher<sensor_msgs::msg::Image>("/sf_transport/depth",qos); auto rp=node->create_publisher<sensor_msgs::msg::Image>("/sf_transport/rgb",qos);
   auto until=clock_type::now()+std::chrono::seconds(15);
   while(dp->get_subscription_count()==0||rp->get_subscription_count()==0){if(clock_type::now()>=until)throw std::runtime_error("subscriber discovery timeout");std::this_thread::sleep_for(std::chrono::milliseconds(10));}
   sensor_msgs::msg::Image d,r;for(auto* im:{&d,&r}){im->width=1280;im->height=720;im->step=5120;im->header.frame_id="synthetic_optical";im->data.resize(size_t(im->step)*im->height,42);}d.encoding="32FC1";r.encoding="rgba8";
   std::vector<Pub> out;out.reserve(count); auto start=clock_type::now();auto epoch=std::chrono::duration_cast<ns>(start.time_since_epoch()).count();
   for(int i=-warmup*60;i<count;++i){auto target=epoch+(int64_t(i)+warmup*60)*period/60;std::this_thread::sleep_until(clock_type::time_point(ns(target)));auto t=wall();if(i>=0&&t>target+period/60){out.push_back({i+1,target,0,0,0,0,0});continue;}
    d.header.stamp.sec=r.header.stamp.sec=i>=0?i+1:-1;d.header.stamp.nanosec=r.header.stamp.nanosec=0;
    if(i>=0){write_payload(d.data.data(),d.data.size(),i+1,'d');write_payload(r.data.data(),r.data.size(),i+1,'r');}
    auto a=wall(),c=cpu();dp->publish(d);auto b=wall(),e=cpu();rp->publish(r);auto f=wall(),g=cpu();if(i>=0)out.push_back({i+1,target,a,b-a,e-c,f-b,g-e});
   }
   std::ofstream file(path);file<<"id,target_ns,begin_ns,depth_wall_ns,depth_cpu_ns,rgb_wall_ns,rgb_cpu_ns\n";for(auto v:out)file<<v.id<<','<<v.target<<','<<v.begin<<','<<v.dwall<<','<<v.dcpu<<','<<v.rwall<<','<<v.rcpu<<'\n';if(!file)throw std::runtime_error("write failure");
  }else{
   std::vector<Rec> out;out.reserve(size_t(count)*2+1000);uint64_t overflow=0,invalid=0;
   auto accept=[&](sensor_msgs::msg::Image::ConstSharedPtr im,char k){if(im->header.stamp.sec==-1)return;if(im->width!=1280||im->height!=720||im->step!=5120||im->data.size()!=3686400||im->header.stamp.sec<1||im->encoding!=(k=='d'?"32FC1":"rgba8")||!valid_payload(im->data.data(),im->data.size(),im->header.stamp.sec,k)){++invalid;return;}if(out.size()==out.capacity()){++overflow;return;}out.push_back({im->header.stamp.sec,wall(),k});};
   auto ds=node->create_subscription<sensor_msgs::msg::Image>("/sf_transport/depth",qos,[&](sensor_msgs::msg::Image::ConstSharedPtr p){accept(p,'d');});auto rs=node->create_subscription<sensor_msgs::msg::Image>("/sf_transport/rgb",qos,[&](sensor_msgs::msg::Image::ConstSharedPtr p){accept(p,'r');});
   rclcpp::executors::SingleThreadedExecutor executor;executor.add_node(node);auto deadline=clock_type::now()+std::chrono::seconds(duration+warmup+8);while(rclcpp::ok()&&clock_type::now()<deadline)executor.spin_once(std::chrono::milliseconds(10));
   std::ofstream file(path);file<<"id,arrival_ns,kind\n";for(auto v:out)file<<v.id<<','<<v.at<<','<<v.kind<<'\n';file.close();std::cout<<"trace_overflow="<<overflow<<" invalid_messages="<<invalid<<"\n";if(!file||overflow||invalid)throw std::runtime_error("subscriber trace failure or overflow");
  }
  rclcpp::shutdown();return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';if(rclcpp::ok())rclcpp::shutdown();return 1;}
}
