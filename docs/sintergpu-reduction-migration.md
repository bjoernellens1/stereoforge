# Experimental SinterGPU reduction migration

`VulkanStereo` can opt in to the SinterGPU 0.2.11 `common.block_reduce` helper
by setting `SINTERGPU_SHARED_REDUCTION=1` before engine construction and
installing the `sintergpu` extra. The extra pins the immutable release wheel by
SHA-256. The runtime also checks the expected package version and released
helper digest before creating a device, then composes the exact helper bytes
into the shader source. `shader_source_identity` records the composed source
hash and helper digest. The original reduction remains the default.

The shared path preserves the 256-thread uint-min tree and caller-owned shared
storage and barriers. It does not add a device, allocation, submission or
readback. Only the Python `VulkanStereo` SGM aggregate path is changed; native
Daxa implementations are separate. CPU tree/model and dual-compiler resource
reflection tests do not establish GPU output parity, timing or application
quality. The old 0.1.0 migration PR has historical Radeon fixture evidence,
but this changed consumer/release compilation needs fresh source-matched GPU
and controlled no-regression qualification before adoption.

A dedicated GPU runner must be independently reviewed for device identity,
exclusive locking, synchronization and teardown before any GPU parity run.

Standalone Slang 2026.12/2026.19 compile and reflect all five entry points
for both sources when entry points are discovered automatically. Explicit
`-entry select` does not resolve on the unchanged baseline in either compiler;
automatic discovery preserves the `select` entry name without altering source.
SlangPy runtime execution of the full pipeline still needs qualification.
