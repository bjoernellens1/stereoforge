#include <stereoforge/daxa_view_calibration.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace stereoforge { namespace {
void require(bool b,const char* m){if(!b)throw std::invalid_argument(m);}
using Matrix=std::array<double,16>;
Matrix mul(Matrix const& a,Matrix const& b){Matrix out{};for(unsigned i=0;i<4;++i)for(unsigned j=0;j<4;++j)for(unsigned k=0;k<4;++k)out[i*4+j]+=a[i*4+k]*b[k*4+j];return out;}
Matrix transpose_rotation(std::array<double,9> const& r){Matrix out{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)out[i*4+j]=r[j*3+i];return out;}
std::array<double,3> rotate(Matrix const& m,std::array<double,3> const& q,bool transpose=false){std::array<double,3> out{};for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)out[i]+=(transpose?m[j*4+i]:m[i*4+j])*q[j];return out;}
void rigid(Pose4 const& p){for(float x:p)require(std::isfinite(x),"finite virtual pose required");for(unsigned i=0;i<4;++i)require(std::abs(p[12+i]-(i==3?1:0))<1e-5,"affine virtual pose required");for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j){double dot=0;for(unsigned k=0;k<3;++k)dot+=double(p[k*4+i])*p[k*4+j];require(std::abs(dot-(i==j?1:0))<1e-4,"rigid virtual pose required");}require(p[0]*(p[5]*p[10]-p[6]*p[9])-p[1]*(p[4]*p[10]-p[6]*p[8])+p[2]*(p[4]*p[9]-p[5]*p[8])>0,"proper virtual rotation required");}
std::array<double,2> distort(DaxaCameraModel const& c,double x,double y){return detail::daxa_calibration_distort(c,x,y);}

} // namespace
DaxaViewCalibration prepare_daxa_view_calibration(DaxaRectification const& c,GeometryConfig config,VirtualView const& view){
    c.validate();require(c.geometry.has_value(),"map geometry association required");auto const& g=*c.geometry;g.validate();require(config.source_width==c.width&&config.source_height==c.height,"rectified source dimensions mismatch");
    uint64_t n=uint64_t(c.width)*c.height,raw=uint64_t(c.eye_width)*c.source_height,m=uint64_t(config.target_width)*config.target_height;require(n&&raw&&m&&n<=UINT32_MAX/2&&raw<=UINT32_MAX/2&&m<=UINT32_MAX/2,"view namespace size overflow");
    rigid(view.source_to_virtual_left);rigid(view.source_to_virtual_right);require(view.projection==Projection::Perspective||view.projection==Projection::Cylindrical,"unknown virtual projection");require(view.mode==ViewMode::Strict||view.mode==ViewMode::Preview,"unknown virtual mode");require(std::isfinite(view.intrinsics.fx)&&view.intrinsics.fx>0&&std::isfinite(view.intrinsics.fy)&&view.intrinsics.fy>0&&std::isfinite(view.intrinsics.cx)&&std::isfinite(view.intrinsics.cy)&&std::isfinite(view.hfov_radians)&&view.hfov_radians>0&&view.hfov_radians<6.28318530718,"invalid virtual camera");
    Matrix v{};std::copy(view.source_to_virtual_left.begin(),view.source_to_virtual_left.end(),v.begin());auto right=mul(v,g.original.right_to_left);for(unsigned i=0;i<16;++i)require(std::abs(right[i]-view.source_to_virtual_right[i])<=1e-6+1e-5*std::abs(right[i]),"original eye virtual transforms disagree with calibrated pose");
    g.validate_map_association({c.maps[0],c.maps[1],c.maps[2],c.maps[3]});
    DaxaViewCalibration out;out.original_width=c.eye_width;out.original_height=c.source_height;out.effective_view=view;out.rays.resize(size_t(2*n));out.preview_indices.assign(size_t(2*m),UINT32_MAX);
    for(unsigned eye=0;eye<2;++eye){auto const& original=g.original.eyes[eye];auto const& rect=g.rectified.eyes[eye];require(original.width==c.eye_width&&original.height==c.source_height&&rect.width==c.width&&rect.height==c.height,"original/rectified calibration dimensions mismatch");auto rt=transpose_rotation(eye?g.R2:g.R1),effective=mul(eye?right:v,rt);auto& pose=eye?out.effective_view.source_to_virtual_right:out.effective_view.source_to_virtual_left;for(unsigned i=0;i<16;++i){pose[i]=float(effective[i]);require(std::isfinite(pose[i]),"effective pose exceeds float range");}
        for(uint32_t y=0;y<c.height;++y)for(uint32_t x=0;x<c.width;++x){size_t i=size_t(y)*c.width+x;std::array<double,3> ray{(x-rect.cx)/rect.fx,(y-rect.cy)/rect.fy,1};out.rays[eye*n+i]={float(ray[0]),float(ray[1]),1};}
        auto transform=eye?right:v;
        for(uint32_t y=0;y<config.target_height;++y)for(uint32_t x=0;x<config.target_width;++x){std::array<double,3> q;
            if(view.projection==Projection::Perspective)q={(double(x)-view.intrinsics.cx)/view.intrinsics.fx,(double(y)-view.intrinsics.cy)/view.intrinsics.fy,1};else{double angle=((double(x)+.5)/config.target_width-.5)*view.hfov_radians;q={std::sin(angle),(double(y)-view.intrinsics.cy)/view.intrinsics.fy,std::cos(angle)};}
            auto direction=rotate(transform,q,true);if(direction[2]<=0)continue;auto uv=distort(original,direction[0]/direction[2],direction[1]/direction[2]);if(uv[0]<-.5||uv[0]>=double(c.eye_width)-.5||uv[1]<-.5||uv[1]>=double(c.source_height)-.5)continue;auto sx=uint32_t(std::floor(uv[0]+.5)),sy=uint32_t(std::floor(uv[1]+.5));out.preview_indices[eye*m+size_t(y)*config.target_width+x]=uint32_t(eye*raw+uint64_t(sy)*c.eye_width+sx);++out.preview_valid_count;}
    }return out;
}
}
