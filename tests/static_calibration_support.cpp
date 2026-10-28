#include "stereoforge/static_calibration_support.hpp"
#include <cmath>
#include <cfenv>
#include <limits>
#include <stdexcept>
#include <vector>
#include <random>
using namespace stereoforge;
#define CHECK(expr) do { if (!(expr)) throw std::runtime_error("check failed: " #expr); } while (false)
// Independent scalar oracle evaluates only mathematically nonzero taps.
static bool oracle(float mx, float my, int w, int h) {
  if (!std::isfinite(mx) || !std::isfinite(my) || std::abs(mx)>32768 || std::abs(my)>32768) return false;
  const int qx=int(std::nearbyint(mx*32.f)), qy=int(std::nearbyint(my*32.f));
  const int x=int(std::floor(qx/32.0)), y=int(std::floor(qy/32.0));
  const int nx=(qx-x*32 != 0) ? 2 : 1, ny=(qy-y*32 != 0) ? 2 : 1;
  return x>=0 && y>=0 && x+nx<=w && y+ny<=h;
}
int main() {
  CHECK(std::fesetround(FE_TONEAREST) == 0);
  CHECK(std::fegetround() == FE_TONEAREST);
  std::array<std::vector<float>,4> maps;
  for (auto &m: maps) m.resize(49);
  for (int y=0;y<7;++y) for(int x=0;x<7;++x) for(int eye=0;eye<2;++eye) { maps[2*eye][y*7+x]=float(x); maps[2*eye+1][y*7+x]=float(y); }
  auto masks=derive_static_calibration_support(7,7,7,7,maps);
  for (int eye=0;eye<2;++eye) for(int y=0;y<7;++y) for(int x=0;x<7;++x) { CHECK(masks.sampling[eye][y*7+x]); CHECK(bool(masks.census5x5[eye][y*7+x]) == (x>=2&&x<5&&y>=2&&y<5)); }
  maps[0][3*7+3]=6.015625f; // exact half rounds to even, no out-of-bounds tap
  masks=derive_static_calibration_support(7,7,7,7,maps); CHECK(masks.sampling[0][24]);
  maps[0][24]=6.015626f; masks=derive_static_calibration_support(7,7,7,7,maps); CHECK(!masks.sampling[0][24]); CHECK(masks.sampling[1][24]);
  for(int y=1;y<=5;++y)for(int x=1;x<=5;++x) if(x>=2&&x<5&&y>=2&&y<5) CHECK(!masks.census5x5[0][y*7+x]);
  maps[0][24]=-0.015625f; masks=derive_static_calibration_support(7,7,7,7,maps); CHECK(masks.sampling[0][24]);
  maps[0][24]=-0.015626f; masks=derive_static_calibration_support(7,7,7,7,maps); CHECK(!masks.sampling[0][24]);
  for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), 32769.f, -32769.f}) {
    for (auto &map : maps) {
      float saved=map[24]; map[24]=bad;
      bool invalid=false; try { derive_static_calibration_support(7,7,7,7,maps); } catch(const std::invalid_argument&) { invalid=true; }
      CHECK(invalid); map[24]=saved;
    }
  }
  bool rejected=false; try { derive_static_calibration_support(0,7,7,7,maps); } catch(const std::invalid_argument&) { rejected=true; } CHECK(rejected);
  maps[3].pop_back(); rejected=false; try { derive_static_calibration_support(7,7,7,7,maps); } catch(const std::invalid_argument&) { rejected=true; } CHECK(rejected);
  // Independent oracle exercises both eyes, arbitrary hole patterns and ties.
  std::mt19937 gen(7);
  for (int w: {1,3,8}) for (int h: {1,4,9}) {
    constexpr int ow=9, oh=7;
    std::array<std::vector<float>,4> random_maps;
    for(auto &m: random_maps) { for(int i=0;i<ow*oh;++i) m.push_back((int(gen()%512)-128)/64.f); }
    auto actual=derive_static_calibration_support(w,h,ow,oh,random_maps);
    for(int eye=0;eye<2;++eye) for(int y=0;y<oh;++y) for(int x=0;x<ow;++x) {
      const int i=y*ow+x;
      CHECK(bool(actual.sampling[eye][i])==oracle(random_maps[eye*2][i],random_maps[eye*2+1][i],w,h));
      bool expected=x>=2 && x+2<ow && y>=2 && y+2<oh;
      if(expected) for(int dy=-2;dy<=2;++dy) for(int dx=-2;dx<=2;++dx) {
        int j=(y+dy)*ow+x+dx;
        expected &= oracle(random_maps[eye*2][j],random_maps[eye*2+1][j],w,h);
      }
      CHECK(bool(actual.census5x5[eye][i])==expected);
    }
  }
  std::array<std::vector<float>,4> tiny; for(auto &m:tiny)m.assign(3,0.f); auto t=derive_static_calibration_support(1,1,3,1,tiny); for(auto v:t.census5x5[0]) CHECK(!v);
}
