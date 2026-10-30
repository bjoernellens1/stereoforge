#ifndef SF_STEREO_FEATURE_ATLAS
#define SF_STEREO_FEATURE_ATLAS

// Input is uint32 luma (0..255) and 0/1 support from the existing
// GPU rectification stage. Result is four uint32 values, 16 bytes per pixel:
// {Census24, Sobel L1, 8-neighbour texture L1, 5x5-support-valid}.
uint4 stereo_atlas_pixel(uint* gray,uint* support,uint width,uint height,uint x,uint y){
    uint4 invalid=uint4(0,0,0,0);
    if(x<2||y<2||x+2>=width||y+2>=height)return invalid;
    [unroll]for(int dy=-2;dy<=2;++dy)
        [unroll]for(int dx=-2;dx<=2;++dx){
            uint index=uint(int(y)+dy)*width+uint(int(x)+dx);
            if(support[index]!=1||gray[index]>255)return invalid;
        }
    uint index=y*width+x,center=gray[index],bits=0,bit=0;
    [unroll]for(int dy=-2;dy<=2;++dy)
        [unroll]for(int dx=-2;dx<=2;++dx){
            if(dx==0&&dy==0)continue;
            uint value=gray[uint(int(y)+dy)*width+uint(int(x)+dx)];
            bits|=uint(value<center)<<(bit++);
        }
    uint tl=gray[(y-1)*width+x-1],t=gray[(y-1)*width+x],tr=gray[(y-1)*width+x+1];
    uint l=gray[y*width+x-1],r=gray[y*width+x+1];
    uint bl=gray[(y+1)*width+x-1],b=gray[(y+1)*width+x],br=gray[(y+1)*width+x+1];
    int gx=int(tr+2*r+br)-int(tl+2*l+bl);
    int gy=int(bl+2*b+br)-int(tl+2*t+tr);
    uint texture=uint(abs(int(tl)-int(center))+abs(int(t)-int(center))+
        abs(int(tr)-int(center))+abs(int(l)-int(center))+
        abs(int(r)-int(center))+abs(int(bl)-int(center))+
        abs(int(b)-int(center))+abs(int(br)-int(center)));
    return uint4(bits,uint(abs(gx)+abs(gy)),texture,1);
}
#endif
