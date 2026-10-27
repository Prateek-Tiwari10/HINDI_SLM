import numpy as np
import pytest

from data.quality import LABEL_NEG, LABEL_POS, check_scores, p_hq


def test_p_hq_is_complement_for_web_label():
    assert p_hq((LABEL_POS,), np.array([0.9])) == pytest.approx(0.9)
    assert p_hq((LABEL_NEG,), np.array([0.9])) == pytest.approx(0.1)
    assert p_hq((LABEL_POS,), np.array([1.00001])) == 1.0


def test_check_scores_stops_on_confidence_like_scores():
    names = ["wikipedia-0000-00", "sangraha_verified-0000-00"]
    good = {names[0]: np.array([0.9, 0.95]), names[1]: np.linspace(0.0, 1.0, 100)}
    check_scores(good, names, ["wikipedia"])
    bad = {names[0]: np.array([0.9, 0.95]), names[1]: np.linspace(0.5, 1.0, 100)}
    with pytest.raises(SystemExit):
        check_scores(bad, names, ["wikipedia"])


def test_random_scores_are_reproducible_and_uniform():
    from data.quality import random_scores
    cfg = {"quality": {"seed": 7}}
    meta = {"a-0001-00": np.ones(50_000), "a-0002-00": np.ones(10)}
    first, again = random_scores(cfg, list(meta), meta), random_scores(cfg, list(meta), meta)
    assert np.array_equal(first["a-0001-00"], again["a-0001-00"])
    assert not np.array_equal(first["a-0001-00"][:10], first["a-0002-00"])  # each shard has its own stream
    assert 0.0 <= first["a-0001-00"].min() and first["a-0001-00"].max() < 1.0
    assert abs(float(np.mean(first["a-0001-00"] < 0.2)) - 0.2) < 0.01


def test_each_method_has_its_own_cache_and_markers():
    from data.pipeline import final_markers_dir
    from data.quality import drop_list_name
    fasttext, rand = {"quality": {"method": "fasttext"}}, {"quality": {"method": "random"}}
    assert drop_list_name(fasttext) == "quality" and final_markers_dir(fasttext) == "final_markers/"
    assert drop_list_name(rand) == "quality_random" and final_markers_dir(rand) == "final_markers_random/"
