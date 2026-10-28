# Publisher thread CPU timing

Pass `--thread-cpu-timing` to `stereoforge_ros2_depth_publisher` to collect
publisher-thread CPU clock samples around each output's host copy and ROS
publish call. It is off by default. Enabling it adds four
`CLOCK_THREAD_CPUTIME_ID` reads per output and changes the time being measured;
use a matched run with the option off when assessing delivery performance.

Each enabled publication has an available sample or an explicit unavailable
reason (`clock_read_failed` or `nonmonotonic`). A clock failure does not stop the
publisher. Summaries report the count of available and failed samples and
percentiles for the available samples. The existing wall-clock durations are
retained. Thread CPU and wall clocks have unrelated epochs: only differences
within each clock are meaningful. A gap between CPU and wall duration does not
identify the cause of time spent off the thread. Older traces without this
option remain readable and report timing as not enabled.
