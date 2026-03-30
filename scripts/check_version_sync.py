import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def extract_first(pattern: str, text: str, file_label: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find version in {file_label}")
    return match.group(1)


def main() -> int:
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    settings_text = (ROOT / "config" / "settings.yaml").read_text(encoding="utf-8")
    dashboard_text = (ROOT / "pgloop" / "visualization" / "dashboard.py").read_text(
        encoding="utf-8"
    )

    pyproject_version = extract_first(
        r'^version\s*=\s*"([^"]+)"',
        pyproject_text,
        "pyproject.toml",
    )
    settings_version = extract_first(
        r'^\s*version:\s*"([^"]+)"',
        settings_text,
        "config/settings.yaml",
    )
    dashboard_version = extract_first(
        r"PG-LCA-TEA v([0-9]+\.[0-9]+\.[0-9]+)",
        dashboard_text,
        "pgloop/visualization/dashboard.py",
    )

    versions = {
        "pyproject.toml": pyproject_version,
        "config/settings.yaml": settings_version,
        "pgloop/visualization/dashboard.py": dashboard_version,
    }

    unique_versions = set(versions.values())
    if len(unique_versions) == 1:
        version = unique_versions.pop()
        print(f"Version sync check passed: {version}")
        return 0

    print("Version mismatch detected:")
    for file_name, version in versions.items():
        print(f"- {file_name}: {version}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
