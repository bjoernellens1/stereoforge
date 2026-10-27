# Experimental shared reduction

The default shader retains the original fixed tree. Set `SINTERGPU_SHARED_REDUCTION=1`
before constructing the first engine to compile the shared SinterGPU fixed256 tree.
This is an experimental opt in pending cross vendor qualification; it changes no
runtime/device ownership or transfer path. Restart the process when changing this
setting.

Install this project's `shared-reduction` extra with Python 3.11+. It pins the
[immutable 0.1.0 wheel](https://github.com/bjoernellens1/sintergpu/releases/download/v0.1.0/sintergpu-0.1.0-py3-none-any.whl#sha256=2de6610de431abb4ed687793dd368ebf09567ebabda5ad5e0f6512cc85c300bf) by exact SHA256.
The shader source release commit is `b90e642b60c6283abec2fb8b505c940617ff3938`.
No PyPI publication is assumed.

Compiled source and helper SHA256 values are recorded at setup. The composed shader includes the exact released helper bytes; both enter its cache identity.
