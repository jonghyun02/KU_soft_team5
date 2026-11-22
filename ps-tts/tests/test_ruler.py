"""길이 목표 시험. 영어 음절 수는 가짜 함수로 바꿔서 eSpeak 없이 돈다."""
import pytest

from pstts import ruler
from pstts.config import Config
from pstts.records import LineItem

FAKE_COUNTS = {
    "I'll pay half.": 3,
    "Where were you last night?": 7,
    "I didn't order two tickets.": 8,
    "": 0,
}


@pytest.fixture(autouse=True)
def fake_english_count(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []

    def count(text: str) -> int:
        seen.append(text)
        return FAKE_COUNTS[text]

    monkeypatch.setattr(ruler, "english_syllable_count", count)
    return seen


def test_default_tolerance_is_about_two_syllables() -> None:
    # 기본 coef 1.28: 26프레임 x (256/22050)초 x (6.19 x 1.28)음절/초 = 2.39 -> 2
    assert Config().tolerance() == 2
    # coef 1.2 였을 때: 2.24 -> 2
    assert Config(coef=1.2).tolerance() == 2


def test_relative_uses_english_syllables_times_coef(fake_english_count: list[str]) -> None:
    cfg = Config(coef=1.2)
    t = ruler.target_for(LineItem("a", "Where were you last night?"), cfg)
    assert (t.n, t.tol, t.en_syllables, t.basis) == (8, 2, 7, "relative")   # 7 x 1.2 = 8.4
    assert t.seconds_est == pytest.approx(8 / cfg.ko_rate)
    assert fake_english_count == ["Where were you last night?"]


def test_relative_rounds_half_up() -> None:
    t = ruler.target_for(LineItem("a", "I'll pay half."), Config(coef=1.5))    # 3 x 1.5 = 4.5
    assert t.n == 5


def test_relative_never_below_one() -> None:
    t = ruler.target_for(LineItem("a", ""), Config())
    assert (t.n, t.en_syllables, t.low) == (1, 0, 1)


def test_explicit_target_wins_over_everything() -> None:
    item = LineItem("a", "I'll pay half.", target_syllables=6, seconds=3.0)
    t = ruler.target_for(item, Config(length_basis="seconds"))
    assert (t.n, t.basis, t.en_syllables) == (6, "explicit", 3)


def test_seconds_basis_uses_korean_rate() -> None:
    cfg = Config(length_basis="seconds", coef=1.2)
    t = ruler.target_for(LineItem("a", "I didn't order two tickets.", seconds=1.5), cfg)
    assert (t.n, t.basis) == (11, "seconds")     # 1.5초 x 7.428 음절/초 = 11.14
    assert t.seconds_est == pytest.approx(11 / cfg.ko_rate)


def test_seconds_ignored_when_basis_is_relative() -> None:
    t = ruler.target_for(LineItem("a", "I didn't order two tickets.", seconds=1.5), Config(coef=1.2))
    assert (t.n, t.basis) == (10, "relative")    # 8 x 1.2 = 9.6


def test_seconds_basis_without_seconds_falls_back_to_relative() -> None:
    t = ruler.target_for(LineItem("a", "I'll pay half."), Config(length_basis="seconds", coef=1.2))
    assert (t.n, t.basis) == (4, "relative")     # 3 x 1.2 = 3.6


def test_tolerance_override_and_bounds() -> None:
    t = ruler.target_for(LineItem("a", "I'll pay half.", target_syllables=2), Config(tol_syllables=3))
    assert (t.tol, t.low, t.high) == (3, 1, 5)


@pytest.mark.parametrize(
    "item,cfg",
    [
        (LineItem("a", "I'll pay half.", target_syllables=0), Config()),
        (LineItem("a", "I'll pay half.", seconds=0.0), Config(length_basis="seconds")),
        (LineItem("a", "I'll pay half."), Config(length_basis="frames")),
        (LineItem("a", "I'll pay half."), Config(tol_syllables=-1)),
    ],
)
def test_bad_settings_raise(item: LineItem, cfg: Config) -> None:
    with pytest.raises(ValueError):
        ruler.target_for(item, cfg)
