struct Parameters {uint64_t raw[6];uint64_t full[8];uint64_t coarse[6];uint stride[6];uint eye_width;uint source_height;uint width;uint height;};
struct Push {uint64_t parameters;};[[vk::push_constant]] Push pc;
#define p (*(Parameters*)pc.parameters)
uint byte_at(uint plane,uint offset){uint* words=(uint*)p.raw[plane];return (words[offset>>2]>>((offset&3)*8))&255;}
uint rgb_at(uint plane,uint x,uint y){uint offset=y*p.stride[plane]+3*x;return byte_at(plane,offset)|(byte_at(plane,offset+1)<<8)|(byte_at(plane,offset+2)<<16)|0xff000000u;}
uint gray15(uint c){return ((c&255)*9798+((c>>8)&255)*19235+((c>>16)&255)*3735+16384)>>15;}
[shader("compute")][numthreads(8,8,1)]
void pack_original(uint3 t:SV_DispatchThreadID){if(t.z>=2||t.x>=p.eye_width||t.y>=p.source_height)return;uint* dst=(uint*)p.full[t.z];dst[t.y*p.eye_width+t.x]=rgb_at(t.z,t.x,t.y);}
[shader("compute")][numthreads(8,8,1)]
void pack_rectified(uint3 t:SV_DispatchThreadID){if(t.z>=2||t.x>=p.width||t.y>=p.height)return;uint i=t.y*p.width+t.x;uint c=rgb_at(2+t.z,t.x,t.y);uint* rgb=(uint*)p.full[2+t.z];uint* luma=(uint*)p.full[4+t.z];uint* support=(uint*)p.full[6+t.z];rgb[i]=c;luma[i]=gray15(c);support[i]=byte_at(4+t.z,t.y*p.stride[4+t.z]+t.x)==1?1:0;}
[shader("compute")][numthreads(8,8,1)]
void coarse_box(uint3 t:SV_DispatchThreadID){uint w=p.width/2,h=p.height/2;if(t.z>=2||t.x>=w||t.y>=h)return;uint* input=(uint*)p.full[2+t.z];uint* mask=(uint*)p.full[6+t.z];uint3 sum=0;bool valid=true;[unroll]for(uint k=0;k<4;++k){uint j=(t.y*2+(k>>1))*p.width+t.x*2+(k&1);uint c=input[j];sum+=uint3(c&255,(c>>8)&255,(c>>16)&255);valid=valid&&mask[j]==1;}sum=(sum+2)/4;uint c=sum.x|(sum.y<<8)|(sum.z<<16)|0xff000000u;uint* rgb=(uint*)p.coarse[t.z];uint* luma=(uint*)p.coarse[2+t.z];uint* support=(uint*)p.coarse[4+t.z];uint i=t.y*w+t.x;rgb[i]=c;luma[i]=gray15(c);support[i]=valid?1:0;}
