import json
from pathlib import Path

_DATA = None


def _load_data():
    global _DATA
    if _DATA is not None:
        return _DATA
    data_path = Path(__file__).resolve().parent / "json" / "data.json"
    with data_path.open("r", encoding="utf-8") as f:
        _DATA = json.load(f)
    return _DATA


def get_profile(name: str) -> str:
    data = _load_data()
    s = ""
    for i in data.get(name, {}):
        st = (
            '<dt class = "basicInfo-item name" >'
            + str(i)
            + '         <dd class = "basicInfo-item value" >'
            + str(data[name][i])
            + "</dd >"
        )
        s += st
    return s
