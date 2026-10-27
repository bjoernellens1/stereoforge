#include "../ops/stereo_output.slang"
struct MetadataParameters {
    uint64_t owner_address;
    uint64_t valid_address;
    uint64_t source_ids_address;
    uint64_t source_ages_address;
    uint64_t output_ids_address;
    uint64_t output_ages_address;
    uint count;
};
struct Push { uint64_t parameters; };
[[vk::push_constant]] ConstantBuffer<Push> pc;
#define p (*(MetadataParameters*)pc.parameters)
#define owners ((uint*)p.owner_address)
#define validity ((uint*)p.valid_address)
#define source_ids ((uint2*)p.source_ids_address)
#define source_ages ((uint2*)p.source_ages_address)
#define output_ids ((uint2*)p.output_ids_address)
#define output_ages ((uint2*)p.output_ages_address)
[shader("compute")]
[numthreads(64,1,1)]
void metadata_fused(uint3 tid:SV_DispatchThreadID)
{
    uint t=tid.x; if(t>=p.count)return;
    uint index=owners[t]; bool present=metadata_has_owner(validity[t],index,p.count);
    uint2 id=uint2(0),age=uint2(0);
    if(present){ id=source_ids[index]; age=source_ages[index]; }
    output_ids[t]=metadata_source_id(present,id);
    output_ages[t]=metadata_source_age(present,age);
}
[shader("compute")]
[numthreads(64,1,1)]
void metadata_ids(uint3 tid:SV_DispatchThreadID)
{
    uint t=tid.x; if(t>=p.count)return;
    uint index=owners[t]; bool present=metadata_has_owner(validity[t],index,p.count);
    uint2 value=uint2(0); if(present)value=source_ids[index];
    output_ids[t]=metadata_source_id(present,value);
}
[shader("compute")]
[numthreads(64,1,1)]
void metadata_ages(uint3 tid:SV_DispatchThreadID)
{
    uint t=tid.x; if(t>=p.count)return;
    uint index=owners[t]; bool present=metadata_has_owner(validity[t],index,p.count);
    uint2 value=uint2(0); if(present)value=source_ages[index];
    output_ages[t]=metadata_source_age(present,value);
}
