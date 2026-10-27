#include <stereoforge/daxa_calibration.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>
namespace stereoforge {
namespace {
void require(bool value,const char* message){if(!value)throw std::invalid_argument(message);}
// Rotation/pinhole and baseline-normalized translation/length comparisons.
// Affine bottom rows and structural projection zeros have separate policies.
bool close(double actual,double expected){return std::isfinite(actual)&&std::isfinite(expected)&&std::abs(actual-expected)<=1e-6+1e-5*std::abs(expected);}
void rotation(const std::array<double,9>& r){
    for(double v:r)require(std::isfinite(v),"finite rectification rotation required");
    for(unsigned row=0;row<3;++row)for(unsigned col=0;col<3;++col){double dot=0;for(unsigned k=0;k<3;++k)dot+=r[k*3+row]*r[k*3+col];require(close(dot,row==col?1.:0.),"calibration rotation must be orthonormal");}
    double det=r[0]*(r[4]*r[8]-r[5]*r[7])-r[1]*(r[3]*r[8]-r[5]*r[6])+r[2]*(r[3]*r[7]-r[4]*r[6]);require(close(det,1.),"calibration rotation determinant must be +1");
}
double pose(const std::array<double,16>& t){
    for(double v:t)require(std::isfinite(v),"finite calibration transform required");
    for(unsigned i=0;i<4;++i){double expected=i==3?1.:0.;require(std::abs(t[12+i]-expected)<=1e-8+1e-5*std::abs(expected),"invalid calibration affine bottom row");}
    std::array<double,9> r{};for(unsigned row=0;row<3;++row)for(unsigned col=0;col<3;++col)r[row*3+col]=t[row*4+col];rotation(r);
    double baseline=std::hypot(t[3],t[7],t[11]);require(std::isfinite(baseline)&&baseline>0,"finite positive calibration baseline required");return baseline;
}
void camera(const DaxaCameraModel& c){require(c.width&&c.height,"positive camera dimensions required");require(std::isfinite(c.fx)&&std::isfinite(c.fy)&&std::isfinite(c.cx)&&std::isfinite(c.cy)&&c.fx>0&&c.fy>0,"finite positive camera focal lengths required");for(double d:c.distortion)require(std::isfinite(d),"finite camera distortion required");}
void stereo(const DaxaStereoCalibration& c){for(auto const& eye:c.eyes)camera(eye);require(c.eyes[0].width==c.eyes[1].width&&c.eyes[0].height==c.eyes[1].height,"stereo eye dimensions mismatch");}
void projection(const std::array<double,12>& p,const DaxaCameraModel& c,double baseline,bool right){
    double translation=right?-c.fx*baseline:0;
    if(right)require(std::isfinite(translation)&&translation<0,"P2 metric translation must be finite and strictly negative");
    std::array<double,12> expected{c.fx,0,c.cx,translation,0,c.fy,c.cy,0,0,0,1,0};
    for(unsigned i=0;i<12;++i){
        // Fourth-column structural zeros are exact. Compare P2 Tx by its
        // dimensionless ratio, so tiny pixel-metres cannot vanish in abs1e-6.
        if(i==3&&right)require(p[i]<0&&close(p[i]/translation,1),"P2 translation differs from calibrated baseline");
        else if(i%4==3)require(p[i]==0,"projection structural translation must be exactly zero");
        else require(close(p[i],expected[i]),"projection matrix differs from rectified pinhole");
    }
}
}
void DaxaRectifiedGeometry::validate()const{
    require(!original_identity.empty()&&!rectified_identity.empty(),"original and rectified calibration identities required");
    stereo(original);stereo(rectified);double baseline=pose(original.right_to_left),rect_baseline=pose(rectified.right_to_left);rotation(R1);rotation(R2);
    auto const& t=rectified.right_to_left;
    require(t[3]>0&&close(rect_baseline/baseline,1),"rectified baseline differs from original positive baseline");
    for(unsigned row=0;row<3;++row)for(unsigned col=0;col<3;++col)require(close(t[row*4+col],row==col?1.:0.),"rectified pose rotation must be identity");
    require(close(t[3]/baseline,1)&&close(t[7]/baseline,0)&&close(t[11]/baseline,0),"rectified pose must have horizontal positive x baseline");
    for(unsigned row=0;row<3;++row){double translated=0;for(unsigned k=0;k<3;++k)translated+=R1[row*3+k]*(original.right_to_left[k*4+3]/baseline);require((row!=0||translated>0)&&close(translated,row==0?1.:0.),"R1 translation is inconsistent with rectified baseline");
        for(unsigned col=0;col<3;++col){double combined=0;for(unsigned a=0;a<3;++a)for(unsigned b=0;b<3;++b)combined+=R1[row*3+a]*original.right_to_left[a*4+b]*R2[col*3+b];require(close(combined,row==col?1.:0.),"R1/R2 rotations disagree with original stereo pose");}}
    auto const& left=rectified.eyes[0];auto const& right=rectified.eyes[1];
    require(close(right.fx,left.fx)&&close(right.fy,left.fy)&&close(right.cy,left.cy),"rectified epipolar intrinsics incompatible");
    for(auto const& eye:rectified.eyes)for(double d:eye.distortion)require(d==0,"rectified distortion must be zero");
    projection(P1,left,0,false);projection(P2,right,rect_baseline,true);
}
MetricCalibration DaxaRectifiedGeometry::metric_calibration()const{
    validate();double baseline=std::hypot(rectified.right_to_left[3],rectified.right_to_left[7],rectified.right_to_left[11]);
    double offset=rectified.eyes[0].cx-rectified.eyes[1].cx;
    auto representable=[](double value){double magnitude=std::abs(value);return std::isfinite(value)&&magnitude<=double(std::numeric_limits<float>::max())&&(value==0||magnitude>=double(std::numeric_limits<float>::denorm_min()));};
    require(representable(rectified.eyes[0].fx)&&representable(baseline)&&representable(offset),"calibration overflows or underflows float metric representation");
    MetricCalibration metric{float(rectified.eyes[0].fx),float(baseline),float(offset)};metric.validate();return metric;
}
}
