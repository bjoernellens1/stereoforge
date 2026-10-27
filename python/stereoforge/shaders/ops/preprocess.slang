// OpenCV4.14 qualified uint8 contract: YUYV20 -> saturated RGB -> RGBremap -> gray15.
uint preprocess_rgb(uint y,uint u,uint v){int yy=max(int(y)-16,0)*1220542,uu=int(u)-128,vv=int(v)-128;uint r=uint(clamp((yy+1673527*vv+524288)>>20,0,255));uint g=uint(clamp((yy-409993*uu-852492*vv+524288)>>20,0,255));uint b=uint(clamp((yy+2116026*uu+524288)>>20,0,255));return r|(g<<8)|(b<<16)|0xff000000u;}
int preprocess_quantize(float coordinate){float scaled=coordinate*32.0,base=floor(scaled),fraction=scaled-base;int value=int(base);return value+int(fraction>0.5||(fraction==0.5&&((value&1)!=0)));}
uint preprocess_gray(uint rgb){return (((rgb&255)*9798+((rgb>>8)&255)*19235+((rgb>>16)&255)*3735+16384)>>15);}
