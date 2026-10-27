struct Push {
    uint64_t resident;
    uint64_t counters;
    uint frame;
    uint phase;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;
[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    uint* resident = (uint*)pc.resident;
    uint* counters = (uint*)pc.counters;
    if (pc.phase == 0) {
        resident[0] = pc.frame;
        resident[1] = pc.frame * 17 + 3;
    } else {
        counters[pc.frame - 1] = (resident[0] == pc.frame && resident[1] == pc.frame * 17 + 3) ? pc.frame : 0;
    }
}
