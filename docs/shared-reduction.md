# Experimental shared reduction

The default shader retains the original fixed tree. Set `SINTERGPU_SHARED_REDUCTION=1`
before constructing the first engine to compile the shared SinterGPU fixed256 tree.
This is an experimental opt in pending cross vendor qualification; it changes no
runtime/device ownership or transfer path. Restart the process when changing this
setting (native engines are cached).

Install the verified SinterGPU wheel from its explicit immutable release wheel URL
once published, then install this project's `shared-reduction` extra. The version
constraint describes compatibility only; it does not claim public PyPI availability.
The release wheel URL and SHA256 must be locked by the integrating release.

Compiled source and helper SHA256 values are recorded at setup. Native HIP staging
contains the exact composed shader and helper bytes, and both enter the cache key.
