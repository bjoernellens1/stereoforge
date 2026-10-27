StructuredBuffer<float> current_depth, current_confidence, current_disparity;
StructuredBuffer<uint> current_valid, current_provenance;
StructuredBuffer<uint2> current_ids;
StructuredBuffer<float> previous_depth, previous_confidence, previous_disparity;
StructuredBuffer<uint> previous_valid, previous_provenance;
StructuredBuffer<uint2> previous_ids;
RWStructuredBuffer<uint> zbuffer, owner, output_valid, output_provenance, choose_previous;
RWStructuredBuffer<float> output_depth, output_confidence, output_disparity;
uniform uint width,height;
uniform float fx,fy,cx,cy;
uniform float4x4 pose;
uniform uint metric_disparity;
uniform float disparity_fx, baseline_m, principal_offset, tolerance;
bool project(uint i,out uint target,out float z) {
 target=0;z=0;
 if(previous_valid[i]==0) return false;
 float d=previous_depth[i];
 if(!isfinite(d)||d<=0) return false;
 float3 p=float3((float(i%width)-cx)/fx,(float(i/width)-cy)/fy,1)*d;
 float4 q=mul(pose,float4(p,1));
 if(!all(isfinite(q))||q.z<=0) return false;
 float u=fx*q.x/q.z+cx,v=fy*q.y/q.z+cy;
 if(!isfinite(u)||!isfinite(v)||u<-.5||v<-.5||u>=float(width)-.5||v>=float(height)-.5) return false;
 target=uint(floor(v+.5))*width+uint(floor(u+.5));z=q.z;return true;
}
[shader("compute")][numthreads(64,1,1)]
void clear(uint3 tid:SV_DispatchThreadID) {
 uint i=tid.x;if(i>=width*height)return;
 zbuffer[i]=0x7f800000;owner[i]=0xffffffff;output_depth[i]=asfloat(0x7fc00000);
 output_disparity[i]=asfloat(0x7fc00000);output_confidence[i]=0;output_valid[i]=0;output_provenance[i]=0;
}
[shader("compute")][numthreads(64,1,1)]
void splat_depth(uint3 tid:SV_DispatchThreadID) {
 uint i=tid.x;if(i>=width*height)return;uint t;float z;uint ignored;
 if(project(i,t,z))InterlockedMin(zbuffer[t],asuint(z),ignored);
}
[shader("compute")][numthreads(64,1,1)]
void splat_owner(uint3 tid:SV_DispatchThreadID) {
 uint i=tid.x;if(i>=width*height)return;uint t;float z;uint ignored;
 if(project(i,t,z)&&asuint(z)==zbuffer[t])InterlockedMin(owner[t],i,ignored);
}
[shader("compute")][numthreads(64,1,1)]
void resolve(uint3 tid:SV_DispatchThreadID) {
 uint i=tid.x;if(i>=width*height)return;uint src=owner[i];if(src==0xffffffff)return;
 float z=asfloat(zbuffer[i]);output_depth[i]=z;output_valid[i]=1;output_confidence[i]=previous_confidence[src];
 uint p=previous_provenance[src];output_provenance[i]=(p==1||p==2)?2:p;
 if(metric_disparity!=0)output_disparity[i]=disparity_fx*baseline_m/z+principal_offset;
}
[shader("compute")][numthreads(64,1,1)]
void fuse(uint3 tid:SV_DispatchThreadID) {
 uint i=tid.x;if(i>=width*height)return;choose_previous[i]=0;
 float z=current_depth[i],c=current_confidence[i],d=current_disparity[i];
 uint valid=current_valid[i],p=current_provenance[i];
 if(previous_valid[i]!=0) {
  if(valid==0){z=previous_depth[i];c=previous_confidence[i];d=asfloat(0x7fc00000);valid=1;
   uint pp=previous_provenance[i];p=(pp==1||pp==2)?2:pp;choose_previous[i]=1;}
  else if(abs(z-previous_depth[i])<=tolerance*z && any(current_ids[i]!=previous_ids[i]) &&
     (p==1||p==2)&&(previous_provenance[i]==1||previous_provenance[i]==2) && c+previous_confidence[i]>0){
    z=(z*c+previous_depth[i]*previous_confidence[i])/(c+previous_confidence[i]);
    c=max(c,previous_confidence[i]);d=asfloat(0x7fc00000);p=3;
  }
 }
 output_depth[i]=z;output_confidence[i]=c;output_disparity[i]=d;output_valid[i]=valid;output_provenance[i]=p;
}
