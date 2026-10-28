"""Guard the diagnostic's boundaries; GPU execution is a separate gate."""
from pathlib import Path

SOURCE = (Path(__file__).resolve().parents[1] / "apps/daxa_gpu_consumer_check.cpp").read_text()


def test_exact_both_planes_and_submission_lifetime():
    compare = SOURCE.split("void compare_exact(", 1)[1].split("\n}\n}", 1)[0]
    assert "DaxaObservationPlane::Depth" in compare
    assert "DaxaObservationImage::RectifiedLeft" in compare
    assert "consumer.depth()" in compare and "consumer.rgba()" in compare
    assert compare.count("std::memcmp(") == 2
    assert compare.index("consumer.retain_until(ticket)") < compare.index("d.wait_on_submit(")
    assert compare.index("d.wait_on_submit(") < compare.index("std::memcmp(")


def test_retention_exhaustion_and_reuse_are_checked():
    main = SOURCE.split("int main(", 1)[1]
    for assertion in ("!consumer->has_capacity()", "consumer->has_capacity()",
                      "consumer->enqueue(second_observation)", "c.depth()==freed_depth",
                      "compare_exact(*context,second_observation,b)"):
        assert assertion in main
    assert main.count("compare_exact(*context,second_observation,b)") == 2
    assert "source2->packed_yuyv[i]^=0x55" in main
    assert "source3->packed_yuyv[i]^=0xaa" in main
    assert "c.source()->frame_id==source3->frame_id" in main
    assert main.index("a={};first_observation={};first={}") < main.index("auto third=pipeline->enqueue")


def test_diagnostic_is_synthetic_validation_only():
    assert 'check(options.validation' in SOURCE
    assert 'device_identity.find("synthetic")' in SOURCE
    assert '{"performance_qualified",false}' in SOURCE
    assert '{"foreign_context_checked",false}' in SOURCE
    assert 'context->validation_errors==0&&context->validation_warnings==0' in SOURCE
    assert SOURCE.index("pipeline.reset()") < SOURCE.index("context->close()")
