// Binding-independent metadata semantics. Signed 64-bit values are transported
// as two little-endian words without arithmetic, truncation or precision loss.
#ifndef SF_STEREO_OUTPUT_SLANG
#define SF_STEREO_OUTPUT_SLANG
// A valid flag does not authorize an out-of-range physical-pointer load.
// Source count is part of the shared contract for both fused and split passes.
bool metadata_has_owner(uint valid, uint owner_index, uint source_count)
{
    return valid != 0 && owner_index != 0xffffffff && owner_index < source_count;
}
uint2 metadata_source_id(bool present, uint2 value)
{
    return present ? value : uint2(0xffffffff, 0xffffffff);
}
uint2 metadata_source_age(bool present, uint2 value)
{
    return present ? value : uint2(0, 0);
}
#endif
