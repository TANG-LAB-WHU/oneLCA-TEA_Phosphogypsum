"""
DataHub Management Module (Deep Module)

Provides centralized, robust, and absolute path management for the multimodal DataHub.
Eliminates hardcoded relative path strings across the entire codebase.
"""

from pathlib import Path
from typing import Optional, Union


class DataHub:
    """
    Centralized path navigation and directory lifecycle manager for DataHub.
    Anchored reliably to the project root directory regardless of current working directory.
    """

    # Project root is 3 levels up from pgloop/iodata/datahub.py
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
    ROOT: Path = PROJECT_ROOT / "datahub"

    # Top-level Medallion tiers
    CACHE_DIR: Path = ROOT / "cache"
    RAW_DIR: Path = ROOT / "raw"
    INTERIM_DIR: Path = ROOT / "interim"
    PROCESSED_DIR: Path = ROOT / "processed"
    TEMPLATES_DIR: Path = ROOT / "templates"

    # Specific functional directories
    RAW_PAPERS_UNPARSED: Path = RAW_DIR / "papers" / "unparsed"
    RAW_EXPERIMENTAL: Path = RAW_DIR / "experimental"
    RAW_TELEMETRY: Path = RAW_DIR / "telemetry"

    INTERIM_PAPERS_PARSED: Path = INTERIM_DIR / "papers" / "parsed"

    PROCESSED_EXTRACTED_DATA: Path = PROCESSED_DIR / "extracted_data"
    PROCESSED_KNOWLEDGE_GRAPH: Path = PROCESSED_DIR / "knowledge_graph"
    PROCESSED_LIGHTRAG_DB: Path = PROCESSED_DIR / "lightrag_db"
    PROCESSED_RAGANYTHING_DB: Path = PROCESSED_DIR / "raganything_db"
    PROCESSED_PARAMETER_RANGES: Path = PROCESSED_DIR / "parameter_ranges"
    PROCESSED_DYNAMIC_ASSESSMENT: Path = PROCESSED_DIR / "dynamic_assessment"
    TELEMETRY_DB: Path = PROCESSED_DIR / "telemetry.db"

    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all primary directories in DataHub exist on disk."""
        dirs = [
            cls.CACHE_DIR,
            cls.RAW_DIR,
            cls.RAW_PAPERS_UNPARSED,
            cls.RAW_EXPERIMENTAL,
            cls.RAW_TELEMETRY,
            cls.INTERIM_DIR,
            cls.INTERIM_PAPERS_PARSED,
            cls.PROCESSED_DIR,
            cls.PROCESSED_EXTRACTED_DATA,
            cls.PROCESSED_KNOWLEDGE_GRAPH,
            cls.PROCESSED_LIGHTRAG_DB,
            cls.PROCESSED_RAGANYTHING_DB,
            cls.PROCESSED_PARAMETER_RANGES,
            cls.PROCESSED_DYNAMIC_ASSESSMENT,
            cls.TEMPLATES_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_template_path(cls, filename: str) -> Path:
        """Get the absolute path to an Excel/CSV template."""
        path = cls.TEMPLATES_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Template '{filename}' not found in {cls.TEMPLATES_DIR}")
        return path

    @classmethod
    def resolve_path(cls, relative_or_absolute: Union[str, Path]) -> Path:
        """
        Safely resolve a path. If given as a relative path under datahub/,
        anchors it to DataHub.ROOT.
        """
        p = Path(relative_or_absolute)
        if p.is_absolute():
            return p
        if str(p).startswith("datahub/") or str(p).startswith("./datahub/"):
            clean_rel = str(p).lstrip("./").removeprefix("datahub/")
            return cls.ROOT / clean_rel
        return (cls.PROJECT_ROOT / p).resolve()
