struct Parameters {
    uint64_t depth,validity,rgba,rays,points,export_points;
    uint width,height,export_width,export_height,stride,pad;
};
struct Point {float x,y,z;uint rgb;};
struct Push {uint64_t parameters;};
[[vk::push_constant]] Push pc;
#define p (*(Parameters*)pc.parameters)
[shader("compute")][numthreads(64,1,1)] void generate(uint3 tid:SV_DispatchThreadID) {
    uint i=tid.x;if(i>=p.width*p.height)return;
    float z=((float*)p.depth)[i];uint c=((uint*)p.rgba)[i];float2 ray=((float2*)p.rays)[i];
    Point point;point.rgb=((c&255u)<<16)|(c&0xff00u)|((c>>16)&255u);
    float2 xy=ray*z;
    bool valid=((uint*)p.validity)[i]==1&&isfinite(z)&&z>0&&all(isfinite(xy));
    point.x=valid?xy.x:asfloat(0x7fc00000u);
    point.y=valid?xy.y:asfloat(0x7fc00000u);
    point.z=valid?z:asfloat(0x7fc00000u);
    ((Point*)p.points)[i]=point;
}
[shader("compute")][numthreads(64,1,1)] void sample_points(uint3 tid:SV_DispatchThreadID) {
    uint i=tid.x;if(i>=p.export_width*p.export_height)return;
    uint source=(i/p.export_width)*p.stride*p.width+(i%p.export_width)*p.stride;
    ((uint4*)p.export_points)[i]=((uint4*)p.points)[source];
}
