"""Exact rational oracle for the bounded Vulkan confidence-division contract.

This helper does not change compound subpixel or RHI/GPU exact comparison gates.
"""
from fractions import Fraction
from pathlib import Path
import random
import struct
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]


def fraction32(bits):
    sign=-1 if bits>>31 else 1
    exponent=(bits>>23)&255;mantissa=bits&0x7fffff
    if exponent==255:return None
    if exponent:mantissa|=1<<23;power=exponent-150
    else:power=-149
    return sign*Fraction(mantissa)*(Fraction(2)**power)


def exact_ulp(q):
    """Locate representable neighbors, then use their gap enclosing exact q."""
    center=struct.unpack('<I',struct.pack('<f',float(q)))[0]
    value=fraction32(center)
    if value==q:
        return min(q-fraction32(center-1),fraction32(center+1)-q)
    if value<q:
        lower,upper=center,center+1
    else:
        lower,upper=center-1,center
    while fraction32(lower)>q:lower-=1
    while fraction32(upper)<q:upper+=1
    return fraction32(upper)-fraction32(lower)


def oracle(bits,numerator,denominator):
    if not 1<=numerator<=denominator<=160000:return False
    value=fraction32(bits)
    if value is None:return False
    q=Fraction(numerator,denominator)
    return 2*abs(value-q)<=5*exact_ulp(q)


@pytest.fixture(scope='module')
def helper(tmp_path_factory):
    directory=tmp_path_factory.mktemp('sgm-float-contract')
    source=directory/'check.cpp';binary=directory/'check'
    source.write_text('''#include <stereoforge/sgm_float_contract.hpp>
#include <bit>
#include <cstdint>
#include <iostream>
int main(){uint32_t bits,n,d;while(std::cin>>std::hex>>bits>>std::dec>>n>>d)
std::cout<<stereoforge::confidence_within_vulkan_division(std::bit_cast<float>(bits),n,d)<<'\\n';}
''')
    subprocess.run(['c++','-std=c++20','-O2','-Wall','-Wextra','-Werror','-I',str(ROOT/'include'),str(source),'-o',str(binary)],check=True)
    def batch(cases):
        text=''.join(f'{bits:x} {n} {d}\n' for bits,n,d in cases)
        result=subprocess.run([str(binary)],input=text,capture_output=True,text=True,check=True)
        return [line=='1' for line in result.stdout.splitlines()]
    return batch


def test_exact_binade_boundaries_use_smaller_adjacent_gap(helper):
    cases=[];expected=[]
    for n,d,center in ((1,1,0x3f800000),(1,2,0x3f000000),(1,4,0x3e800000),(160000,160000,0x3f800000)):
        for offset,allowed in ((0,True),(1,True),(2,False),(-1,True),(-2,True),(-3,False)):
            cases.append((center+offset,n,d));expected.append(allowed)
    assert [oracle(*case) for case in cases]==expected
    assert helper(cases)==expected


def test_actual_seven_over_sixty_three_gpu_bits_are_inside_normative_bound(helper):
    case=(0x3de38e38,7,63)
    assert oracle(*case)
    assert helper([case])==[True]


def test_invalid_domains_nonfinite_and_coarse_range_are_rejected(helper):
    cases=[(bits,n,d) for bits in (0,0x80000000,0x3f800000,0x3de38e38)
        for n,d in ((0,1),(1,0),(2,1),(1,160001),(0xffffffff,0xffffffff))]
    cases += [(bits,1,2) for bits in (0x7f800000,0xff800000,0x7fc00000,0x7f800001,0xffc00000,0xbf000000,0x35ffffff,0x40000001)]
    assert not any(oracle(*case) for case in cases)
    assert helper(cases)==[False]*len(cases)


def test_fraction_oracle_matches_normal_neighbors_and_shift_extremes(helper):
    rng=random.Random(0x53474d)
    ratios=[(1,160000),(159999,160000),(160000,160000),(1,131072)]
    ratios += [(1,1<<power) for power in range(18)]
    for _ in range(2500):
        d=rng.randint(1,160000);ratios.append((rng.randint(1,d),d))
    cases=[]
    for n,d in ratios:
        center=struct.unpack('<I',struct.pack('<f',float(Fraction(n,d))))[0]
        cases.extend((center+offset,n,d) for offset in range(-6,7))
        cases.extend((bits,n,d) for bits in (0x36000000,0x40000000,rng.randint(0x36000000,0x40000000)))
    expected=[oracle(*case) for case in cases]
    assert any(expected) and not all(expected)
    assert helper(cases)==expected
