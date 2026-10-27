#pragma once
#include <bit>
#include <cstdint>
#include <limits>

namespace stereoforge {
// Only the single, non-RelaxedPrecision binary32 confidence division n/d.
// Vulkan's 2.5-ULP bound is centered on exact n/d, never the CPU-rounded result:
// https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#spirvenv-precision-operation
// At an exact power of two, ulp is the smaller adjacent representable gap.
// Invalid pixels require an independent exact-zero check by the caller. This
// utility does not permit any change to compound float or RHI/GPU byte gates.
inline bool confidence_within_vulkan_division(float g,uint32_t numerator,uint32_t denominator) {
    static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559 &&
                  std::numeric_limits<float>::digits==24);
    if(numerator==0 || denominator==0 || numerator>denominator || denominator>160000)return false;
    const uint32_t bits=std::bit_cast<uint32_t>(g);
    // Positive finite normal g in [2^-19,2]. Anything else cannot be within
    // 2.5 ULP of the permitted exact q in [1/160000,1]. Bit tests also reject
    // both infinities, every NaN, negative values, signed zero and subnormals.
    if(bits<0x36000000u || bits>0x40000000u)return false;

    int e=0;
    while((uint64_t(numerator)<<unsigned(-e))<denominator)--e;
    const bool exact_power=(uint64_t(numerator)<<unsigned(-e))==denominator;
    const int u=e-(exact_power?24:23); // ulp(q)=2^u, using the exact binade.
    const uint32_t m=(bits&0x007fffffu)|0x00800000u;
    const int e2=int((bits>>23)&0xffu)-150; // g=m*2^e2 exactly.
    const int k=(-e2>-u)?-e2:-u;

    // Bounds: -18<=e<=0, -41<=u<=-24, -42<=e2<=-22,
    // k<=42, 0<=e2+k<=20 and 0<=u+k<=18. With m<2^24
    // and d<2^18, scaled_g<2^62; scaled_n<2^60. The absolute
    // difference is <2^62, its double <2^63, and RHS<2^38.
    // Thus every shift/product below is exact uint64 arithmetic without wrap.
    const uint64_t scaled_g=(uint64_t(m)*denominator)<<unsigned(e2+k);
    const uint64_t scaled_n=uint64_t(numerator)<<unsigned(k);
    const uint64_t error=scaled_g>scaled_n?scaled_g-scaled_n:scaled_n-scaled_g;
    return 2ull*error <= (5ull*denominator)<<unsigned(u+k);
}
}
