#pragma once
#include <stereoforge/daxa_preprocess.hpp>
namespace stereoforge {
// One captured YUYV upload/decode with independently calibrated output grids.
// Grids must share packed source identity and shape. Consumers such as refinement
// must independently validate their stricter 2x geometry and map association.
// A shared source identity does not attest map or camera-model geometry.
class DaxaDualPreprocess {
    struct Impl; std::shared_ptr<Impl> impl_;
public:
    struct Frame { DaxaFrame full, coarse; };
    DaxaDualPreprocess(DaxaRectification full, DaxaRectification coarse, unsigned slots, std::shared_ptr<DaxaFoundation>);
    Frame enqueue(CaptureLease);
    size_t collect();
    DaxaPreprocessReadback readback(const DaxaFrame&) const;
    std::shared_ptr<DaxaFoundation> context() const;
};
}
