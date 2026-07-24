"""Configurações centralizadas do sistema de reconhecimento facial."""

from typing import Final, FrozenSet

# ─── Cache ────────────────────────────────────────────────────────────────
CACHE_FILE: Final[str] = "processamento_cache.json"
CACHE_INTERVAL: Final[int] = 1

# ─── Imagem ───────────────────────────────────────────────────────────────
MAX_DIMENSION: Final[int] = 6000

# ─── Reconhecimento facial ───────────────────────────────────────────────
DET_SIZE: Final[tuple[int, int]] = (640, 640)
MODEL_NAME: Final[str] = "buffalo_l"

# buffalo_s produz embeddings com similaridades ligeiramente menores
# que buffalo_l; o threshold 0.35 compensa mantendo precisão similar.
SIMILARITY_THRESHOLD: Final[float] = 0.5

# ─── Extensões de imagem ─────────────────────────────────────────────────
IMAGE_EXTENSIONS: Final[FrozenSet[str]] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
)
RAW_EXTENSIONS: Final[FrozenSet[str]] = frozenset({
    ".nef", ".cr2", ".cr3", ".arw", ".orf",
    ".rw2", ".srw", ".raf", ".dng", ".raw",
})
RAW_PREVIEW_QUALITY: Final[int] = 95
RAW_HALF_SIZE: Final[bool] = True

# ─── Performance / Paralelização ─────────────────────────────────────────
CPU_TARGET_PERCENT: Final[int] = 100
MIN_WORKERS: Final[int] = 3
MAX_WORKERS: Final[int] = 16
PARALLEL_MIN_PHOTOS: Final[int] = 2
