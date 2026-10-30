#include <stereoforge/gpu_duty_budget.hpp>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
using namespace stereoforge;
namespace {
template<class F> bool throws(F f) {try{f();}catch(...){return true;}return false;}
void basic_budget() {
    GpuDutyBudget b(GpuDutyConfig{25, 8'000, 32, 5});
    const auto start=1'000'000'000ull;
    unsigned admitted=0;
    for(std::uint64_t i=0;i<600;++i) {
        const auto time=start+i*1'000'000'000ull/60;
        if(b.may_admit(time)) {
            b.reserve(i+1,time);
            b.complete(i+1,8.0);
            ++admitted;
        }
    }
    // 25% GPU duty at 8ms/frame permits approximately 31.25fps.
    assert(admitted>=30*10 && admitted<=34*10);
    assert(b.stats().reservations==admitted && b.stats().completions==admitted);
    assert(b.stats().denials==600-admitted && b.stats().missing_timings==0);
    assert(b.stats().charged_gpu_ns==uint64_t(admitted)*8'000'000);
}
void out_of_order_and_missing() {
    GpuDutyBudget b(GpuDutyConfig{30, 8'000, 24, 5});
    const auto t=1'000'000'000ull;
    assert(b.may_admit(t));b.reserve(1,t);
    assert(b.may_admit(t+1));b.reserve(2,t+1);
    assert(b.may_admit(t+2));b.reserve(3,t+2);
    assert(!b.may_admit(t+3));
    b.complete(2,std::numeric_limits<double>::quiet_NaN());
    b.complete(3,12.0);
    b.complete(1,5.0);
    assert(b.stats().completions==3 && b.stats().missing_timings==1);
    assert(b.stats().charged_gpu_ns==25'000'000);
    assert(throws([&]{b.complete(2,3.0);}));
    assert(throws([&]{b.reserve(77,t+99);}));
    assert(throws([&]{b.may_admit(t-1);}));
}
void configuration() {
    assert(throws([]{GpuDutyBudget x(GpuDutyConfig{0,8000,33,5});}));
    assert(throws([]{GpuDutyBudget x(GpuDutyConfig{101,8000,33,5});}));
    assert(throws([]{GpuDutyBudget x(GpuDutyConfig{10,8000,2,5});}));
    assert(throws([]{GpuDutyBudget x(GpuDutyConfig{10,8000,33,30});}));
    GpuDutyBudget disabled(GpuDutyConfig{100,8000,33,60});
    assert(!disabled.enabled());
    assert(disabled.may_admit(1));
    disabled.reserve(1,1);
    disabled.complete(1,0);
    assert(disabled.stats().missing_timings==1);
}
void debt_recovery() {
    GpuDutyBudget b(GpuDutyConfig{50,8'000,16,5});
    assert(b.may_admit(1'000'000'000ull));b.reserve(1,1'000'000'000ull);
    b.complete(1,50.0);
    assert(b.stats().max_estimated_debt_ns>0);
    assert(!b.may_admit(1'010'000'000ull));
    assert(b.may_admit(1'110'000'000ull));
}
}
int main(){basic_budget();out_of_order_and_missing();configuration();debt_recovery();}
