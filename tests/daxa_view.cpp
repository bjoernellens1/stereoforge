#include <stereoforge/daxa_view.hpp>
#include <iostream>
#include <cmath>
#include <limits>
#include <stdexcept>
using namespace stereoforge;
void check(bool x,char const* message){if(!x)throw std::runtime_error(message);}
template<class F>void rejected(F f){try{f();}catch(std::exception const&){return;}throw std::runtime_error("expected rejection");}
int main(int argc,char** argv){try{
    const bool cpu=argc==2&&std::string(argv[1])=="--cpu-only";
    GeometryConfig c{2,1,3,1};VirtualView v;v.intrinsics={1,1,0,0};
    std::vector<float> depth{2,2,2,2},confidence(4,.75f);std::vector<uint32_t> rgb{0x102030,0x405060,0x102030,0x405060},prov(4,5),preview{0,1,1,2,3,3};
    std::vector<Ray3> rays{{0,0,1},{1,0,1},{0,0,1},{1,0,1}};
    std::vector<int64_t> ids{(1ll<<40)+1,(3ll<<40)+2,(1ll<<40)+3,(1ll<<40)+4},ages{(1ll<<42)+7,(5ll<<42)+9,(1ll<<42)+7,(5ll<<42)+9};
    GeometrySources source{depth,confidence,rgb,prov,rays,ids,ages,123,"test-calibration",preview};
    DaxaView::validate(c,source,v,1);
    for(size_t component:{size_t(0),size_t(1),size_t(12),size_t(15)}){auto bad=v;bad.source_to_virtual_left[component]+=0.25f;rejected([&]{DaxaView::validate(c,source,bad,1);});}
    auto reflection=v;reflection.source_to_virtual_right[0]=-1;rejected([&]{DaxaView::validate(c,source,reflection,1);});
    rejected([&]{DaxaView::validate({0,1,3,1},source,v,1);});rejected([&]{DaxaView::validate(c,source,v,4);});
    auto missing=source;missing.calibration_id.clear();rejected([&]{DaxaView::validate(c,missing,v,1);});
    auto wrong=source;wrong.confidence={};rejected([&]{DaxaView::validate(c,wrong,v,1);});
    preview[0]=3;rejected([&]{DaxaView::validate(c,source,v,1);});preview[0]=0;
    DaxaView::Frame empty;rejected([&]{empty.wait();});
    if(cpu){std::cout<<"CPU input contracts passed\n";return 0;}
    for(auto mode:{ViewMode::Strict,ViewMode::Preview})for(unsigned ring=1;ring<=3;++ring){
        v.mode=mode;auto engine=std::make_unique<DaxaView>(c,source,v,ring,true);
        std::vector<DaxaView::Frame> held;for(unsigned i=0;i<ring;++i)held.push_back(engine->project());
        rejected([&]{engine->project();}); // GPU completion does not revoke a consumer lease.
        for(auto const& frame:held){check(frame.timestamp_ns()==123&&frame.calibration_id()=="test-calibration"&&frame.config().target_width==3,"frame timestamp/calibration/shape lost");
            frame.wait();auto counter=engine->inspect(frame);
            uint32_t expected_hash=0;for(size_t i=0;i<2;++i)expected_hash^=uint32_t(ids[i])^uint32_t(uint64_t(ids[i])>>32)^uint32_t(ages[i])^uint32_t(uint64_t(ages[i])>>32);
            uint32_t truncated_hash=0;for(size_t i=0;i<2;++i)truncated_hash^=uint32_t(ids[i])^uint32_t(ages[i]);
            check(expected_hash!=truncated_hash,"GPU metadata oracle cancels selected high words");
            check(counter[0]==2&&counter[1]==uint32_t(mode==ViewMode::Preview)&&counter[2]==expected_hash&&counter[3]==0,"downstream GPU count/hash mismatch");
            auto bytes_before=engine->allocated_bytes();auto out=engine->readback(frame);check(bytes_before==engine->allocated_bytes(),"diagnostic readback grew retained memory");
            check(out.depth.size()==3&&out.owner.size()==3,"eleven planes wrong size");
            for(size_t i=0;i<2;++i){check(out.validity[i]==1&&out.depth[i]==2&&out.confidence[i]==.75f,"metric output mismatch");check(out.rgb[i]==rgb[i]&&out.rgb_validity[i]==1&&out.provenance[i]==5&&out.source[i]==3&&out.approximate[i]==0&&out.owner[i]==i,"RGB/metadata output mismatch");check(out.source_ids[i]==ids[i]&&out.age_ns[i]==ages[i],"64-bit metadata truncated");}
            check(out.validity[2]==0&&std::isnan(out.depth[2])&&out.confidence[2]==0&&out.provenance[2]==0&&out.source_ids[2]==-1&&out.age_ns[2]==0,"unsupported hole fabricated geometry");
            check(out.rgb_validity[2]==uint32_t(mode==ViewMode::Preview)&&out.approximate[2]==uint32_t(mode==ViewMode::Preview),"preview mask mismatch");
            check(frame.gpu_ms()>=0,"timestamp invalid");}
        rejected([&]{engine->project();});auto old_generation=held.front().generation();held.clear();
        auto next=engine->project();check(next.generation()>old_generation,"ring generation not advanced");next.wait();
        // Resolution/calibration replacement owns a separate context, old lease stays usable.
        auto other_source=source;other_source.calibration_id="replacement-calibration";
        std::vector<uint32_t> larger_preview{0,1,1,1,2,3,3,3};other_source.preview_indices=larger_preview;
        ids[0]^=1ll<<47;
        auto other=std::make_unique<DaxaView>(GeometryConfig{2,1,4,1},other_source,v,1,true);rejected([&]{other->readback(next);});rejected([&]{other->inspect(next);});ids[0]^=1ll<<47;
        auto other_frame=other->project();other_frame.wait();
        auto mutated=other->inspect(other_frame);uint32_t original_hash=uint32_t((uint64_t(ids[0])>>32)^(uint64_t(ids[1])>>32)^uint32_t(ids[0])^uint32_t(ids[1])^(uint64_t(ages[0])>>32)^(uint64_t(ages[1])>>32)^uint32_t(ages[0])^uint32_t(ages[1]));
        check(mutated[2]==(original_hash^(1u<<15))&&mutated[2]!=original_hash,"GPU consumer ignored original metadata high-word mutation");check(other_frame.context_id()!=next.context_id(),"foreign contexts share identity");other_frame={};
        auto other_diagnostics=other->diagnostics();other.reset();check(other_diagnostics->closed&&other_diagnostics->errors==0&&other_diagnostics->warnings==0,"replacement context teardown validation failed");
        auto diagnostics=engine->diagnostics();engine.reset();next.wait();check(next.buffer("output_source_ids").is_empty()==false,"facade destruction invalidated consumer lease");
        check(!diagnostics->closed,"context closed while consumer retains lease");next={};check(diagnostics->closed&&diagnostics->errors==0&&diagnostics->warnings==0,"context teardown validation failed");
    }
    std::cout<<"Daxa view contracts passed\n";return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
