#pragma once
#include <stereoforge/daxa_observation.hpp>
#include <stereoforge/stereo_trajectory.hpp>
#include <optional>
#include <vector>
namespace stereoforge {
enum class DaxaStereoHistoryPlane:unsigned { Left,Right,Depth,Validity,Provenance,SourceIds,AgeNs };
struct DaxaStereoHistoryConfig {
    unsigned capacity=4;
    uint64_t max_age_ns=200000000,max_bytes=1073741824;
    bool allow_unqualified_research=false;
    void validate() const;
};
struct DaxaStereoHistoryView {daxa::BufferId buffer;uint64_t offset=0,size=0;};
struct DaxaStereoHistoryStats {uint64_t offered=0,copied=0,pose_rejected=0,capacity_skipped=0,expired=0,resets=0;};
// Owns GPU copies: retaining four cached pairs never retains four producers.
// Admission/collection/consumption use the shared device's one submission thread.
class DaxaStereoHistory {
    struct Slot;struct Impl;std::unique_ptr<Impl> impl_;
public:
    class Frame {
        std::shared_ptr<Slot> slot_;uint64_t generation_=0;
        explicit Frame(std::shared_ptr<Slot>);void validate()const;friend class DaxaStereoHistory;
    public:
        Frame()=default;explicit operator bool()const{return bool(slot_);}
        void wait()const;bool ready()const;
        StereoWorldPose const& pose()const;
        daxa::BufferId buffer()const;
        DaxaStereoHistoryView plane(DaxaStereoHistoryPlane)const;
        uint64_t copy_ticket()const;
        void mark_consumed(uint64_t submitted_ticket)const;
    };
    struct Selection {
        Frame frame;uint64_t ingress_age_ns=0;
        std::optional<uint64_t> acquisition_age_upper_bound_ns;
    };
    DaxaStereoHistory(std::shared_ptr<DaxaFoundation>,DaxaObservationFrame const&,DaxaStereoHistoryConfig={});
    ~DaxaStereoHistory();
    // Empty on capacity pressure or unqualified/lost pose. Invalid source throws.
    std::optional<Frame> store(DaxaObservationFrame const&,StereoWorldPose const&);
    std::vector<Selection> select(StereoWorldPose const&)const;
    std::size_t collect();void clear();
    DaxaStereoHistoryStats stats()const;
    uint64_t requested_bytes()const;
    bool compatible(DaxaObservationFrame const&)const;
};
}
