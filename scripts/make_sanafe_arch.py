#!/usr/bin/env python3
"""Write src/sanafe_plugin/build/loihi_delayed.yaml: SANA-FE's bundled loihi.yaml (v2.2.9) with
one extra soma unit, loihi_lif_delayed, whose energy and latency attributes are copied from
loihi_lif and whose model is the plugin src/sanafe_plugin/loihi_lif_delayed.cpp. Nothing else
changes. Run with .venv-neuro after scripts/build_sanafe_plugin.sh."""
from importlib.resources import files
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
B = REPO / "src/sanafe_plugin/build"


def main():
    src = (files("sanafe.examples") / "loihi.yaml").read_text()
    lines = src.splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip() == "- name: loihi_lif")
    ind = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    j = i + 1
    block = []
    while j < len(lines) and (lines[j].startswith(ind + " ") and not lines[j].strip().startswith("- name:")):
        block.append(lines[j]); j += 1
    new = [ind + "- name: loihi_lif_delayed"]
    for l in block:
        if l.strip().startswith("model:"):
            pre = l[: len(l) - len(l.lstrip())]
            new += [pre + "model: loihi_lif_delayed", pre + f"plugin: {B / 'libloihi_lif_delayed.so'}"]
        else:
            new.append(l)
    out = lines[:j] + new + lines[j:]
    (B / "loihi_delayed.yaml").write_text("\n".join(out) + "\n")
    print("\n".join(new))


if __name__ == "__main__":
    main()
