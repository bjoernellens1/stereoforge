# Dual-preprocess live camera smoke evidence

This is exploratory evidence from the attached ZED 2, AMD Radeon 8060S and a ROS 2 Jazzy/`rmw_zenoh_cpp` host-export depth and rectified RGB path. It does not qualify the depth estimator's geometry or establish end-to-end 60 Hz RGB-D performance.

## Dual path: 10-second validation-requested run

The `experimental-dual-refinement` publisher and independent subscriber both exited successfully. The common 10.0001 s window delivered **600 unique depth frames and 600 unique RGB frames**, or **59.9993 Hz** per stream. Publisher ring drops were zero. The capture queue-drop counter increased by zero over the measured publisher window; its final cumulative value was eight, including startup. Arrival gaps and host-ingress-to-subscriber arrival were:

| Measure | Depth | Rectified RGB |
|---|---:|---:|
| Host ingress to arrival, p50 / p99 | 11.005 / 11.960 ms | 11.612 / 12.681 ms |
| Arrival gap, p99 / maximum | 18.169 / 18.956 ms | 18.180 / 18.856 ms |

Host ingress is not camera exposure. Camera clock mapping is unverified and exposure-to-consumer age is unavailable. Host-exported ROS images are not GPU zero-copy. The wide RGB-D view was not included. Ten seconds does not meet the planned sustained 600-second qualification. Producer requested bytes and the aggregate requested total were unavailable; storage accounting is incomplete and no frozen delivery policy was applied. The delivery analyzer therefore marks the run unqualified.

The run requested Vulkan validation. The publisher exited successfully and checks validation error and warning counters through context teardown; no warnings or errors were reported. Synchronization-validation activation was not independently attested.

## Separate-producer comparison attempts

The two subsequent 30-second `experimental-refinement` attempts, `a1` and `a1retry`, each failed before capture with publisher exit code 1 and `unidentified camera serial`; the subscribers exited 0. Neither produced usable performance samples, so there is **no matched old-versus-dual live comparison**. A separate raw V4L attempt requesting 60 frames timed out without delivering a frame after those failures. That observation does not establish whether camera transport, enumeration, a device state, or application setup caused the failures. The user was asked to reconnect the camera before further tests.

## Build and trace provenance

The container build was made from a dirty working tree based on commit `710f4f4ffca944cc9ec8733fee26fe56133d16da`; the publisher itself reports its build commit as unattested. A source-file manifest covering tracked and non-ignored untracked files, including symlink target text, matched before and after the successful run. The 55 recorded executable and SPIR-V checks also passed. These hashes identify the local artifacts; they do not attest separately mounted dependency source trees.

| Artifact | SHA-256 |
|---|---|
| Source-file hash manifest | `c8984bf94fd1f57909d13b3c40dc6f25d770604efc3d78df83f2264f5cf31e53` |
| Binary and SPIR-V hash manifest | `d23d753c1ec26506f6f812a4d6aa9cdbb22bc5e530f9a2abf3f43af7b87cf389` |
| ROS depth publisher | `e05a2eb3cd095bee4ff9c66fe3438a14347da85907cf511f05117e22e5925d3b` |
| ROS delivery probe | `86011ce8a2ddc6377d9acaa26992362d8468ce0e6dcb09783dd1236495309b2f` |
| Container image ID | `459d16995481f8cc5c655bcac5b3bec957b2026032b9a75e780f680b56c25558` |

The raw traces, device properties and private calibration materials remain outside the repository. After camera recovery, repeat a matched old/dual/dual/old run with fresh router and subscriber processes, preserve unique-frame and drop counters, and run longer validation separately. Live imagery differs between runs, so exact numerical parity remains a separate same-input test.
