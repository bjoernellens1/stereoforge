#include "../ops/preprocess.slang"
struct Parameters {uint64_t raw;uint64_t original[2];uint64_t rectified[2];uint64_t luma[2];uint64_t support[2];uint64_t maps[4];uint eye_width;uint source_height;uint stride;uint width;uint height;};
struct Push {uint64_t parameters;};[[vk::push_constant]] Push pc;
#define p (*(Parameters*)pc.parameters)
uint raw_byte(uint offset){uint* words=(uint*)p.raw;return (words[offset>>2]>>((offset&3)*8))&255;}
[shader("compute")][numthreads(8,8,1)]
void decode_yuyv(uint3 tid:SV_DispatchThreadID){if(tid.x>=p.eye_width||tid.y>=p.source_height||tid.z>=2)return;uint pair=tid.y*p.stride+(tid.z*p.eye_width+(tid.x&~1u))*2;uint color=preprocess_rgb(raw_byte(pair+((tid.x&1)!=0?2:0)),raw_byte(pair+1),raw_byte(pair+3));uint* output=(uint*)p.original[tid.z];output[tid.y*p.eye_width+tid.x]=color;}
[shader("compute")][numthreads(8,8,1)]
void rectify_gray(uint3 tid:SV_DispatchThreadID){if(tid.x>=p.width||tid.y>=p.height||tid.z>=2)return;uint i=tid.y*p.width+tid.x;float* mx=(float*)p.maps[tid.z*2];float* my=(float*)p.maps[tid.z*2+1];int qx=preprocess_quantize(mx[i]),qy=preprocess_quantize(my[i]);int x=qx>>5,y=qy>>5;uint fx=uint(qx)&31,fy=uint(qy)&31;uint weights[4]={(32-fx)*(32-fy),fx*(32-fy),(32-fx)*fy,fx*fy};uint3 channels=0;bool full_support=true;uint* original=(uint*)p.original[tid.z];
    [unroll]for(uint tap=0;tap<4;++tap){int xx=x+int(tap&1),yy=y+int(tap>>1);bool inside=xx>=0&&yy>=0&&xx<int(p.eye_width)&&yy<int(p.source_height);if(weights[tap]!=0&&!inside)full_support=false;if(inside){uint rgb=original[uint(yy)*p.eye_width+uint(xx)];channels+=uint3(rgb&255,(rgb>>8)&255,(rgb>>16)&255)*weights[tap];}}
    channels=(channels+512)>>10;uint color=channels.x|(channels.y<<8)|(channels.z<<16)|0xff000000u;uint* rectified=(uint*)p.rectified[tid.z];uint* luma=(uint*)p.luma[tid.z];uint* support=(uint*)p.support[tid.z];rectified[i]=color;luma[i]=preprocess_gray(color);support[i]=full_support?1:0;
}
