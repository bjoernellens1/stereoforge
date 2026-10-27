"""Four-lane selection must preserve scalar shader decisions and arithmetic."""
from pathlib import Path
import re

import numpy as np
import pytest

from stereoforge.reference import select_disparity

ROOT = Path(__file__).resolve().parents[1]
SHADER = ROOT / 'python/stereoforge/shaders/stereo/sgm.slang'


def test_select_uses_four_lanes_with_uniform_barriers():
    source = SHADER.read_text()
    select = source[source.index('void select('):source.index('// Optional metric output.')]
    assert '[numthreads(256,1,1)]' in source[:source.index('void select(')].split('[shader("compute")]')[-1]
    assert 'group_id.x*64+group_thread_id.x/4' in select
    assert 'group_thread_id.x%4' in select
    assert 'd+=4' in select
    assert 'stride=2;stride>0;stride/=2' in select
    assert select.count('GroupMemoryBarrierWithGroupSync()') == 2
    assert 'return' not in select[:select.rindex('GroupMemoryBarrierWithGroupSync()')]
    assert 'if(disparity_count==1)second=best_cost;' in select
    assert 'best>0&&best_cost<10000&&second>best_cost' in select
    python = (ROOT / 'python/stereoforge/gpu.py').read_text()
    assert re.search(r"kernels\['select'\]\.dispatch\(\[n\*4,1,1\]", python)


SCALAR = r'''
StructuredBuffer<uint> sums;
RWStructuredBuffer<float> disparity;
RWStructuredBuffer<float> confidence;
RWStructuredBuffer<uint> validity;
uint width,height,disparity_count,subpixel;
[shader("compute")] [numthreads(64,1,1)]
void scalar(uint3 tid:SV_DispatchThreadID){
    uint i=tid.x; if(i>=width*height)return;
    uint offset=i*disparity_count,best=0,best_cost=0xffffffff,second=0xffffffff;
    for(uint d=0;d<disparity_count;d++){
        uint c=sums[offset+d];
        if(c<best_cost){second=best_cost;best_cost=c;best=d;}else second=min(second,c);
    }
    if(disparity_count==1)second=best_cost;
    bool valid=best>0&&best_cost<10000&&second>best_cost;
    float value=float(best);
    if(subpixel!=0&&best>0&&best+1<disparity_count){
        float lo=float(sums[offset+best-1]),hi=float(sums[offset+best+1]);
        float denominator=lo+hi-2*float(best_cost);
        if(denominator>0)value+=.5*(lo-hi)/denominator;
    }
    disparity[i]=valid?value:asfloat(0x7fc00000);
    confidence[i]=valid?float(second-best_cost)/float(max(second,1u)):0;
    validity[i]=valid?1:0;
}
'''


@pytest.mark.parametrize('D', [1, 2, 3, 31, 32, 64, 128, 256])
def test_four_lane_top_two_merge_invariant(D):
    sentinel = int(np.iinfo(np.uint32).max)
    rng = np.random.default_rng(D)
    rows = [np.full(D, sentinel), np.full(D, 7),
            *rng.integers(0, 12, size=(30, D))]
    # Exercise second minima in one lane and in a different lane.
    for runner_up in [2, 5]:
        if runner_up < D:
            row = np.full(D, 99); row[1] = 3; row[runner_up] = 5
            rows.append(row)
    for row in rows:
        lanes = []
        for lane in range(4):
            best_cost, best_index, second = sentinel, sentinel, sentinel
            for d in range(lane, D, 4):
                cost = int(row[d])
                if (cost, d) < (best_cost, best_index):
                    second, best_cost, best_index = best_cost, cost, d
                else:
                    second = min(second, cost)
            lanes.append((best_cost, best_index, second))
        for stride in [2, 1]:
            for lane in range(stride):
                own, other = lanes[lane], lanes[lane+stride]
                winner, loser = (other, own) if other[:2] < own[:2] else (own, other)
                lanes[lane] = (winner[0], winner[1], min(winner[2], loser[0]))
        expected = sorted((int(cost), d) for d, cost in enumerate(row))
        expected_second = expected[1][0] if D > 1 else expected[0][0]
        best_cost, best_index, second = lanes[0]
        if D == 1:
            second = best_cost
        assert (best_cost, best_index, second) == (*expected[0], expected_second)


@pytest.fixture(scope='module')
def select_kernels():
    s = pytest.importorskip('slangpy')
    device = s.create_device(s.DeviceType.vulkan, enable_cuda_interop=False)
    optimized = device.load_module_from_source('select_reduction', SHADER.read_text())
    scalar = device.load_module_from_source('select_scalar', SCALAR)
    kernels = [device.create_compute_kernel(device.link_program([m], [m.entry_point(e)]))
               for m, e in [(optimized, 'select'), (scalar, 'scalar')]]
    return s, device, kernels


@pytest.mark.gpu
@pytest.mark.parametrize('D', [1, 2, 3, 31, 32, 64, 128, 256])
@pytest.mark.parametrize('subpixel', [False, True])
def test_select_matches_scalar_bits_and_reference(select_kernels, D, subpixel):
    s, device, kernels = select_kernels
    # 67 pixels require a partially inactive final group. Include ties, zero,
    # MAX sentinels, cutoff rejection and second minima in different lanes.
    sums = np.random.default_rng(D).integers(1, 5000, (1, 67, D), dtype=np.uint32)
    sums[0, 0] = np.iinfo(np.uint32).max
    sums[0, 1] = 10000
    sums[0, 2] = 30; sums[0, 2, 0] = 0
    sums[0, 3] = 50; sums[0, 3, min(1, D-1)] = 10
    sums[0, 4] = 40
    sums[0, 5] = 10001
    if D > 1:
        sums[0, 4, 0] = sums[0, 4, D-1] = 5
        sums[0, 5, 1] = 10000
    if D > 2:
        sums[0, 6] = 90; sums[0, 6, 1] = 8; sums[0, 6, 2] = 8
        sums[0, 7] = 70; sums[0, 7, 1] = 5; sums[0, 7, 2] = 11
    if D > 4:
        sums[0, 8] = 99; sums[0, 8, 4] = 5; sums[0, 8, 3] = 7
        sums[0, 9] = 99; sums[0, 9, 1] = 5; sums[0, 9, 4] = 7
    usage = s.BufferUsage.shader_resource | s.BufferUsage.unordered_access
    costs = device.create_buffer(element_count=sums.size, struct_size=4, usage=usage)
    costs.copy_from_numpy(sums.reshape(-1))
    results = []
    for kernel, threads in zip(kernels, [67*4, 67]):
        outputs = {name: device.create_buffer(element_count=67, struct_size=4, usage=usage)
                   for name in ['disparity', 'confidence', 'validity']}
        kernel.dispatch([threads, 1, 1], vars=dict(width=67, height=1,
            disparity_count=D, subpixel=int(subpixel), sums=costs, **outputs))
        results.append({name: b.to_numpy().view(np.uint32).copy() for name, b in outputs.items()})
    for name in results[0]:
        np.testing.assert_array_equal(results[0][name], results[1][name], err_msg=name)
    expected = select_disparity(sums, subpixel=subpixel)
    np.testing.assert_array_equal(results[0]['validity'], expected[2].reshape(-1))
    np.testing.assert_allclose(results[0]['disparity'].view(np.float32), expected[0].reshape(-1), rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(results[0]['confidence'].view(np.float32), expected[1].reshape(-1), rtol=1e-6)
