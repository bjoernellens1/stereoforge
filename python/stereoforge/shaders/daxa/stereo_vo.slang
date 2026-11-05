struct Feature {float u,v,x,y,z;uint valid;};
struct Track {float px,py,pz,cx,cy,cz,pu,pv,cu,cv;uint valid,pad;};
struct Parameters {
    uint64_t current_image,previous_image,depth,validity,provenance,previous_features,current_features,tracks,source_ids,ages;
    uint width,height,columns,rows,has_previous;
    float fx,fy,cx,cy;
    float rig_from_rect[9];
    uint2 source_id;
};
struct Push {uint64_t parameters;};
[[vk::push_constant]] Push pc;
#define p (*(Parameters*)pc.parameters)
float gray(uint64_t image,int x,int y) {
    uint c=((uint*)image)[y*p.width+x];return (float(c&255)*.299+float((c>>8)&255)*.587+float((c>>16)&255)*.114)/255.;
}
bool inside(float2 uv,int margin) {return all(uv>=float(margin))&&uv.x<float(p.width)-margin-1&&uv.y<float(p.height)-margin-1;}
float sample_image(uint64_t image,float2 uv) {
    int2 q=int2(floor(uv));float2 f=uv-float2(q);
    return lerp(lerp(gray(image,q.x,q.y),gray(image,q.x+1,q.y),f.x),lerp(gray(image,q.x,q.y+1),gray(image,q.x+1,q.y+1),f.x),f.y);
}
bool metric(float2 uv,out float3 point) {
    point=0;if(!inside(uv,1))return false;
    int2 q=int2(floor(uv));float2 f=uv-float2(q);float depths[4];
    for(uint k=0;k<4;++k) {
        uint i=(q.y+int(k/2))*p.width+q.x+k%2;
        uint prov=((uint*)p.provenance)[i];float z=((float*)p.depth)[i];
        if(any(((uint2*)p.source_ids)[i]!=p.source_id)||any(((uint2*)p.ages)[i]!=uint2(0,0)))return false;
        if(((uint*)p.validity)[i]==0||(prov!=1&&prov!=5)||!isfinite(z)||z<.1||z>10)return false;
        depths[k]=z;
    }
    float lo=min(min(depths[0],depths[1]),min(depths[2],depths[3]));
    float hi=max(max(depths[0],depths[1]),max(depths[2],depths[3]));
    if(hi-lo>.01*lo)return false;
    float z=lerp(lerp(depths[0],depths[1],f.x),lerp(depths[2],depths[3],f.x),f.y);
    float3 rect=float3((uv.x-p.cx)*z/p.fx,(uv.y-p.cy)*z/p.fy,z);
    for(uint i=0;i<3;++i)point[i]=p.rig_from_rect[i*3]*rect.x+p.rig_from_rect[i*3+1]*rect.y+p.rig_from_rect[i*3+2]*rect.z;
    return all(isfinite(point))&&point.z>0;
}
float corner(int x,int y) {
    float xx=0,xy=0,yy=0;
    for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx) {
        float gx=.5*(gray(p.current_image,x+dx+1,y+dy)-gray(p.current_image,x+dx-1,y+dy));
        float gy=.5*(gray(p.current_image,x+dx,y+dy+1)-gray(p.current_image,x+dx,y+dy-1));
        xx+=gx*gx;xy+=gx*gy;yy+=gy*gy;
    }
    return .5*(xx+yy-sqrt((xx-yy)*(xx-yy)+4*xy*xy));
}
[shader("compute")][numthreads(64,1,1)] void detect(uint3 tid:SV_DispatchThreadID) {
    uint id=tid.x;if(id>=p.columns*p.rows)return;
    Feature outv=(Feature)0;float best=.0004;
    uint col=id%p.columns,row=id/p.columns;
    int x0=max(12,int(col*p.width/p.columns)),x1=min(int(p.width)-13,int((col+1)*p.width/p.columns));
    int y0=max(12,int(row*p.height/p.rows)),y1=min(int(p.height)-13,int((row+1)*p.height/p.rows));
    for(int y=y0;y<y1;y+=2)for(int x=x0;x<x1;x+=2) {
        float score=corner(x,y);if(score<=best)continue;float3 point;
        if(!metric(float2(x,y),point))continue;
        best=score;outv.u=x;outv.v=y;outv.x=point.x;outv.y=point.y;outv.z=point.z;outv.valid=1;
    }
    ((Feature*)p.current_features)[id]=outv;
}
float patch_cost(uint64_t a,uint64_t b,float2 origin,float2 target,int step) {
    if(!inside(origin,step*2+1)||!inside(target,step*2+1))return 1e10;
    float sum=0,square=0;
    for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx) {
        float2 delta=float2(dx,dy)*step;float difference=sample_image(a,origin+delta)-sample_image(b,target+delta);
        sum+=difference;square+=difference*difference;
    }
    return max(0.,(square-sum*sum/25.)/25.);
}
bool match_patch(uint64_t a,uint64_t b,float2 origin,out float2 target) {
    target=origin;float best=1e10;
    for(int level=2;level>=0;--level) {
        int step=1<<level,radius=level==2?4:2;float2 center=target,winner=center;best=1e10;
        for(int dy=-radius;dy<=radius;++dy)for(int dx=-radius;dx<=radius;++dx) {
            float2 candidate=center+float2(dx,dy)*step;float cost=patch_cost(a,b,origin,candidate,step);
            if(cost<best){best=cost;winner=candidate;}
        }
        target=winner;
    }
    if(best>.0025)return false;
    // Mean-subtracted Lucas-Kanade refinement. Unlike a three-cost parabola,
    // an exact photometric match remains an exact stationary point.
    for(int iteration=0;iteration<6;++iteration) {
        if(!inside(target,4))return false;
        float sum_r=0,sum_x=0,sum_y=0,xx=0,xy=0,yy=0,xr=0,yr=0;
        for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx) {
            float2 delta=float2(dx,dy),uv=target+delta;
            float r=sample_image(a,origin+delta)-sample_image(b,uv);
            float gx=.5*(sample_image(b,uv+float2(1,0))-sample_image(b,uv-float2(1,0)));
            float gy=.5*(sample_image(b,uv+float2(0,1))-sample_image(b,uv-float2(0,1)));
            sum_r+=r;sum_x+=gx;sum_y+=gy;xx+=gx*gx;xy+=gx*gy;yy+=gy*gy;xr+=gx*r;yr+=gy*r;
        }
        xx-=sum_x*sum_x/25.;xy-=sum_x*sum_y/25.;yy-=sum_y*sum_y/25.;
        xr-=sum_x*sum_r/25.;yr-=sum_y*sum_r/25.;
        float det=xx*yy-xy*xy;if(det<1e-8)return false;
        float2 offset=float2(yy*xr-xy*yr,xx*yr-xy*xr)/det;
        target+=clamp(offset,-.5,.5);
        if(length(offset)<.005)break;
    }
    return inside(target,4)&&patch_cost(a,b,origin,target,1)<.0025;
}
[shader("compute")][numthreads(64,1,1)] void track(uint3 tid:SV_DispatchThreadID) {
    uint id=tid.x;if(id>=p.columns*p.rows)return;
    Track outv=(Track)0;
    if(p.has_previous==0){((Track*)p.tracks)[id]=outv;return;}
    Feature previous=((Feature*)p.previous_features)[id];
    if(previous.valid!=0) {
        float2 uv=float2(previous.u,previous.v),current,back;float3 point;
        if(match_patch(p.previous_image,p.current_image,uv,current)&&match_patch(p.current_image,p.previous_image,current,back)&&length(back-uv)<=.8&&metric(current,point)) {
            outv.px=previous.x;outv.py=previous.y;outv.pz=previous.z;outv.cx=point.x;outv.cy=point.y;outv.cz=point.z;
            outv.pu=uv.x;outv.pv=uv.y;outv.cu=current.x;outv.cv=current.y;outv.valid=1;
        }
    }
    ((Track*)p.tracks)[id]=outv;
}
