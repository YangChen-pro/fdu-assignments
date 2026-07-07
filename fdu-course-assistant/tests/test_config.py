from datetime import datetime
from pathlib import Path

from fdu_course_assistant.config import load_config, resolve_run_window


def test_load_example_config_targets():
    config = load_config(Path(__file__).parents[1] / "configs" / "example.yaml")
    assert config.targets == ("yjsxk.fudan.edu.cn", "yjsxk.fudan.sh.cn")
    assert config.courses[0].classification_code == 7


def test_single_target_is_supported(tmp_path):
    path = tmp_path / "single-target.yaml"
    path.write_text(
        "target: yjsxk.fudan.edu.cn\n"
        "courses:\n"
        "  - category: 政治理论课\n"
        "    ids:\n"
        "      - C1\n",
        encoding="utf-8",
    )
    assert load_config(path).targets == ("yjsxk.fudan.edu.cn",)


def test_resolve_run_window_rolls_end_to_next_day():
    window = resolve_run_window("23:00:00", "00:30:00", now=datetime(2026, 7, 7, 12, 0, 0))
    assert window.end_at.day == 8
