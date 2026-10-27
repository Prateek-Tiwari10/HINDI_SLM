import pytest

from tokenizer.ladder import choose_fraction, ram_check, should_extend
from tokenizer.sample import band_of, keep_draw, rung_input

LCFG = {"fertility_tol": 0.005, "heavy_split_tol_pp": 0.10, "ram_per_text_byte": 11.0, "ram_headroom": 0.85}
FRACTIONS = [0.05, 0.15, 0.30, 0.50]


def rung(pct, fertility, heavy):
    return {"sample_pct": pct, "fertility": fertility, "heavy_split_pct": heavy}


def test_band_of_edges():
    assert band_of(0.0, FRACTIONS) == 0
    assert band_of(0.0499, FRACTIONS) == 0
    assert band_of(0.05, FRACTIONS) == 1
    assert band_of(0.2999, FRACTIONS) == 2
    assert band_of(0.5, FRACTIONS) is None


def test_rung_shares_match_fractions():
    """Rung k (bands 0..k) holds about fractions[k] of the docs."""
    bands = [band_of(keep_draw(f"doc-{i}", 2024), FRACTIONS) for i in range(20_000)]
    for k, fraction in enumerate(FRACTIONS):
        share = sum(1 for b in bands if b is not None and b <= k) / len(bands)
        assert abs(share - fraction) < 0.015


def test_rung_input_is_cumulative():
    info = {"train_bytes": 1000, "bands": [
        {"fraction": 0.05, "docs": 5, "bytes": 50, "chars": 20, "lines": 7},
        {"fraction": 0.15, "docs": 10, "bytes": 100, "chars": 40, "lines": 14}]}
    files, size = rung_input(["b0", "b1"], info, 1)
    assert files == ["b0", "b1"]
    assert size["bytes"] == 150 and size["lines"] == 21 and size["fraction"] == 0.15
    assert size["pct_of_train_bytes"] == 15.0


def test_smallest_rung_within_tolerance_wins():
    rows = [rung(5, 1.4500, 4.40), rung(15, 1.4440, 4.35), rung(30, 1.4420, 4.33)]
    # 5% is 0.55% worse on fertility (> 0.5%); 15% is 0.14% worse and 0.02 pp on heavy splits.
    assert choose_fraction(rows, LCFG, [])["sample_pct"] == 15


def test_heavy_splits_can_veto():
    rows = [rung(5, 1.4430, 4.60), rung(30, 1.4420, 4.33)]  # fertility fine, heavy splits +0.27 pp
    assert choose_fraction(rows, LCFG, [])["sample_pct"] == 30


def test_flat_curve_picks_smallest():
    rows = [rung(5, 1.4421, 4.34), rung(15, 1.4420, 4.33), rung(30, 1.4420, 4.33)]
    assert choose_fraction(rows, LCFG, [])["sample_pct"] == 5


def test_extend_only_when_still_improving():
    assert should_extend([rung(15, 1.4500, 4.4), rung(30, 1.4400, 4.3)], LCFG)
    assert not should_extend([rung(15, 1.4401, 4.3), rung(30, 1.4400, 4.3)], LCFG)


@pytest.mark.parametrize("text_gb, ram_gb, fits", [(0.95, 31, True), (5.7, 31, False), (5.7, 330, True)])
def test_ram_guard(text_gb, ram_gb, fits):
    assert ram_check(LCFG, text_gb * 1e9, ram_gb * 1e9)[0] is fits
