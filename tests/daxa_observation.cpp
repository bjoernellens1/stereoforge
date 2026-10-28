#include <stereoforge/daxa_observation.hpp>
#include <stereoforge/daxa_depth.hpp>
#include <type_traits>
#include <stdexcept>
#include <iostream>
using namespace stereoforge;
static_assert(std::is_nothrow_move_constructible_v<DaxaObservationFrame>);
static_assert(std::is_nothrow_assignable_v<DaxaObservationFrame&,DaxaObservationFrame>);
static_assert(!std::is_constructible_v<DaxaObservationFrame,std::shared_ptr<DaxaFoundation>>);
template<class F>static void reject(F f){try{f();}catch(std::invalid_argument const&){return;}throw std::runtime_error("empty observation accepted");}
int main(){try{DaxaObservationFrame empty; if(empty)throw std::runtime_error("default observation valid");reject([&]{empty.wait();});reject([&]{empty.ready();});reject([&]{empty.generation();});reject([&]{empty.buffer(DaxaObservationPlane::Depth);});reject([&]{empty.image(DaxaObservationImage::OriginalLeft);});reject([&]{empty.context();});reject([&]{empty.source();});reject([&]{empty.map_owner();});reject([&]{empty.submit_ticket();});DaxaDepthPipeline::Frame depth;reject([&]{depth.observations();});auto copy=empty;auto moved=std::move(copy);moved=moved;moved=std::move(moved);moved=empty;empty={};std::cout<<"Daxa observation CPU contracts passed\n";return 0;}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
