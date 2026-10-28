#include <stereoforge/ros_delivery_trace.hpp>
#include <atomic>
#include <cstdlib>
#include <new>
#include <stdexcept>
using namespace stereoforge;
static std::atomic<unsigned> allocations{0};
void* operator new(std::size_t size) {++allocations;if(auto p=std::malloc(size))return p;throw std::bad_alloc();}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
static void check(bool value){if(!value)throw std::runtime_error("trace contract failed");}
int main(){
    bool rejected=false;try{RosDeliveryTrace trace(1);}catch(const std::invalid_argument&){rejected=true;}
    check(rejected);
    RosDeliveryTrace trace(4); // Two slots reserved for start and stop.
    RosDeliveryTraceRecord a{};a.kind=RosDeliveryTraceKind::Published;a.output=RosDeliveryOutput::Depth;a.source_id=42;a.at_ns=99;
    auto baseline=allocations.load();check(trace.append(a));a.source_id=43;a.at_ns=100;check(trace.append(a));
    check(!trace.append(a));check(!trace.append(a));check(allocations.load()==baseline);
    check(trace.records().size()==2&&trace.records()[0].source_id==42&&trace.records()[1].source_id==43);
    check(trace.dropped()==2&&trace.event_capacity()==4&&trace.requested_bytes()==2*sizeof(RosDeliveryTraceRecord));
}
