from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import re


@dataclass
class DescriptorDB:
    exact: Dict[str, str]
    prefix_numbered: List[Dict[str, str]]  # [{"prefix": str, "desc": str}]
    atomic_suffix: List[Dict[str, str]]   # [{"suffix": str, "desc": str}]

    def get(self, feature: str) -> Optional[str]:
        if feature in self.exact:
            return self.exact[feature]
        for item in self.prefix_numbered:
            p = item["prefix"]
            if feature.startswith(p) and feature[len(p):].isdigit():
                return item["desc"]
            if p == "ES_transition_":
                if re.search(r"(?:^|_)ES(\d+)_transition(?:_|$)", feature):
                    return item["desc"]
            if p == "ES_osc_strength_":
                if re.search(r"(?:^|_)ES(\d+)_osc_strength(?:_|$)", feature):
                    return item["desc"]
        for item in self.atomic_suffix:
            suf = item["suffix"]
            if feature.endswith("_" + suf):
                lead = feature[: -(len(suf)+1)]
                if lead and re.match(r"^[A-Za-z][A-Za-z0-9_]*$", lead):
                    return item["desc"]
        return None


def _parse_descriptor_md(md_text: str) -> DescriptorDB:
    exact: Dict[str, str] = {}
    prefix_numbered: List[Dict[str, str]] = []
    atomic_suffix: List[Dict[str, str]] = []
    for raw_line in md_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.replace("\t", " ").replace("  ", " ")
        m = re.match(r"^([^:]+):\s*(.+)$", line)
        if not m:
            continue
        name = m.group(1).strip()
        desc = m.group(2).strip()
        if name.startswith("<atom>_"):
            suffix = name[len("<atom>_"):]
            atomic_suffix.append({"suffix": suffix, "desc": desc})
            continue
        if name.endswith("[number]"):
            prefix = name[: -len("[number]")]
            prefix_numbered.append({"prefix": prefix, "desc": desc})
            continue
        exact[name] = desc
    return DescriptorDB(exact=exact, prefix_numbered=prefix_numbered, atomic_suffix=atomic_suffix)


def load_descriptor_db(project_root: Path) -> Optional[DescriptorDB]:
    path = project_root / "edbo" / "DESCRIPTOR.md"
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return _parse_descriptor_md(text)
    except Exception:
        return None
