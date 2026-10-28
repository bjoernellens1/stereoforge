#pragma once
// Test-only lookup for a pre-rounded binary32 numerator and offset.
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#if defined(__i386__) || defined(__x86_64__)
#include <xmmintrin.h>
#endif

struct MetricEntry {uint32_t depth,valid;};
static_assert(sizeof(MetricEntry)==8);
struct MetricLut {
    uint32_t version,disparities,depth_convention,numerator_bits,offset_bits;uintptr_t device_identity;std::array<MetricEntry,128> entries;
};
static MetricLut make_metric_lut(float numerator,float offset,uintptr_t device){
    if(std::fegetround()!=FE_TONEAREST||!std::isfinite(numerator)||numerator<std::numeric_limits<float>::min()||!std::isfinite(offset)||(offset!=0&&std::fpclassify(offset)==FP_SUBNORMAL)||!device)
        throw std::runtime_error("unsupported LUT configuration or floating point mode");
#if defined(__i386__) || defined(__x86_64__)
    if((_mm_getcsr() & ((1u<<6)|(1u<<15)))!=0)throw std::runtime_error("CPU DAZ or FTZ enabled");
#else
    throw std::runtime_error("unsupported CPU floating point mode verification");
#endif
    volatile float tiny=std::numeric_limits<float>::denorm_min();volatile float input=tiny;volatile float probe=tiny+input;
    if(std::bit_cast<uint32_t>(float(probe))!=2u)throw std::runtime_error("CPU subnormals unavailable");
    MetricLut table{1,128,1,std::bit_cast<uint32_t>(numerator),std::bit_cast<uint32_t>(offset),device,{}};
    for(uint32_t d=0;d<128;++d){MetricEntry entry{0x7fc00000u,0};if(d){volatile float denominator=float(d)-offset;if(denominator>0){volatile float depth=numerator/denominator;if(std::isfinite(depth)&&depth>0){entry.depth=std::bit_cast<uint32_t>(float(depth));entry.valid=1;}}}table.entries[d]=entry;}
    return table;
}
static void validate_metric_lut(const MetricLut& table,float numerator,float offset,uintptr_t device){
    auto expected=make_metric_lut(numerator,offset,device);
    if(table.version!=expected.version||table.disparities!=expected.disparities||table.depth_convention!=expected.depth_convention||table.numerator_bits!=expected.numerator_bits||table.offset_bits!=expected.offset_bits||table.device_identity!=expected.device_identity)throw std::runtime_error("metric LUT identity mismatch");
    for(size_t i=0;i<table.entries.size();++i)if(table.entries[i].depth!=expected.entries[i].depth||table.entries[i].valid!=expected.entries[i].valid)throw std::runtime_error("metric LUT content mismatch");
}
