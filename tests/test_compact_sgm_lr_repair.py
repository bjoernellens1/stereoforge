import numpy as np
from benchmarks.compact_sgm_lr_repair import repair


def planes(width=12):
    raw = np.full((2, 1, width), np.nan, np.float32)
    valid = np.zeros(raw.shape, bool)
    descriptors = np.zeros(raw.shape, np.uint32)
    support = np.ones(raw.shape, bool)
    return raw, valid, descriptors, support


def test_left_seed_rounds_and_searches_opposite_reference():
    raw, valid, desc, support = planes()
    raw[0, 0, 7], valid[0, 0, 7] = 2.4, True
    desc[1, 0, 5] = 15
    desc[0, 0, :] = 0
    desc[0, 0, 7] = 15
    result = repair(raw, valid, desc, support, 5)
    assert result.raw[1, 0, 5] == 2
    assert result.lr[0, 0, 7] and result.lr[1, 0, 5]
    assert result.repaired[1, 0, 5]


def test_right_seed_sign_and_original_valid_is_never_overwritten():
    raw, valid, desc, support = planes()
    raw[1, 0, 3], valid[1, 0, 3] = 2, True
    raw[0, 0, 5], valid[0, 0, 5] = 4, True
    result = repair(raw, valid, desc, support, 5)
    assert result.raw[0, 0, 5] == 4
    assert not result.repaired.any()
    assert not result.lr[1, 0, 3]


def test_ties_and_unsupported_never_create_depth():
    raw, valid, desc, support = planes()
    raw[0, 0, 7], valid[0, 0, 7] = 2, True
    result = repair(raw, valid, desc, support, 5)
    assert not result.repaired.any()
    assert result.ties == 1
    support[1, 0, 5] = False
    result = repair(raw, valid, desc, support, 5)
    assert not result.repaired.any()
    assert result.unsupported == 1


def test_conflicting_seeds_share_target_without_overwrite():
    raw, valid, desc, support = planes()
    raw[0, 0, 7], valid[0, 0, 7] = 2, True
    raw[0, 0, 8], valid[0, 0, 8] = 3, True
    desc[1, 0, 5] = 15
    desc[0, 0, 7] = 15
    result = repair(raw, valid, desc, support, 5)
    assert result.requests == 2 and result.targets == 1
    assert not result.repaired[1, 0, 5]
    assert result.conflicts == 1
    assert not result.lr[0, 0, 7] and not result.lr[0, 0, 8]
    assert result.raw[0, 0, 8] == 3


def test_occluded_and_out_of_image_seed_stays_invalid():
    raw, valid, desc, support = planes()
    raw[0, 0, 1], valid[0, 0, 1] = 3, True
    result = repair(raw, valid, desc, support, 5)
    assert result.targets == 0
    assert not result.lr.any()


def test_wrong_seed_does_not_force_opposite_winner():
    raw, valid, desc, support = planes()
    raw[0, 0, 7], valid[0, 0, 7] = 2, True
    desc[1, 0, 5] = 15
    desc[0, 0, 9] = 15
    result = repair(raw, valid, desc, support, 6)
    assert result.raw[1, 0, 5] == 4
    assert not result.lr[0, 0, 7]


def test_valid_mutually_agreeing_wrong_matches_are_not_repaired():
    raw, valid, desc, support = planes()
    raw[0, 0, 7], raw[1, 0, 5] = 2, 2
    valid[0, 0, 7] = valid[1, 0, 5] = True
    result = repair(raw, valid, desc, support, 5)
    assert result.lr[0, 0, 7] and result.lr[1, 0, 5]
    assert not result.repaired.any()
    # Agreement is not proof of correctness: truth is intentionally absent.


def test_right_seed_can_repair_left_and_preserves_sign():
    raw, valid, desc, support = planes()
    raw[1, 0, 3], valid[1, 0, 3] = 2, True
    desc[0, 0, 5] = 15
    desc[1, 0, 3] = 15
    result = repair(raw, valid, desc, support, 5)
    assert result.raw[0, 0, 5] == 2
    assert result.lr[0, 0, 5] and result.lr[1, 0, 3]


def test_zero_disparity_competes_and_zero_winner_is_invalid():
    raw, valid, desc, support = planes()
    raw[0, 0, 7], valid[0, 0, 7] = 2, True
    desc[1, 0, 5] = desc[0, 0, 5] = 15
    result = repair(raw, valid, desc, support, 5)
    assert not result.repaired[1, 0, 5]
    assert result.zero_winners == 1


def test_zero_tie_cannot_create_repair():
    raw, valid, desc, support = planes()
    raw[0, 0, 7], valid[0, 0, 7] = 2, True
    desc[1, 0, 5] = desc[0, 0, 5] = desc[0, 0, 7] = 15
    result = repair(raw, valid, desc, support, 5)
    assert not result.repaired[1, 0, 5]
    assert result.ties == 1


def test_random_prior_lr_accepts_are_preserved():
    from stereoforge.refinement import lr_consistency
    rng = np.random.default_rng(83)
    for _ in range(30):
        raw = rng.integers(1, 5, size=(2, 4, 16)).astype(np.float32)
        valid = rng.random(raw.shape) < .45
        desc = rng.integers(0, 256, size=raw.shape, dtype=np.uint32)
        support = rng.random(raw.shape) < .9
        before = lr_consistency(raw, valid, 1.)
        after = repair(raw, valid, desc, support, 5)
        assert np.all(after.lr[before])
        np.testing.assert_array_equal(after.raw[valid], raw[valid])


def test_collisions_on_both_views_are_rejected():
    raw, valid, desc, support = planes(16)
    raw[0, 0, 7], raw[0, 0, 8] = 2, 3
    valid[0, 0, 7] = valid[0, 0, 8] = True
    raw[1, 0, 10], raw[1, 0, 11] = 3, 2
    valid[1, 0, 10] = valid[1, 0, 11] = True
    result = repair(raw, valid, desc, support, 6)
    assert result.conflicts == 2
    assert not result.repaired.any()
