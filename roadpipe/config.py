"""
roadpipe.config
Central configuration for the Route Resilience pipeline (PS04).

Case-study city: Hazaribagh, Jharkhand.
Two preset areas are provided for the real OSM / satellite mode.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

# --------------------------------------------------------------------------
# Case-study city
# --------------------------------------------------------------------------
CITY_NAME = "Hazaribagh"
CITY_STATE = "Jharkhand"
CITY_COUNTRY = "India"

# --------------------------------------------------------------------------
# Preset bounding boxes for Hazaribagh (lat/lon as (north, south, east, west))
# Coordinates approximate the named belts; OSMnx will pull the real network.
# --------------------------------------------------------------------------
# Town center: NH-33 / Matwari / Indrapuri dense grid
HAZARIBAGH_TOWN: Tuple[float, float, float, float] = (24.0080, 23.9830, 85.3760, 85.3490)

# Greater Hazaribagh: town + Hazaribagh Lake + Korra belt (wider window)
HAZARIBAGH_GREATER: Tuple[float, float, float, float] = (24.0300, 23.9550, 85.3950, 85.3250)

AREAS: Dict[str, Tuple[float, float, float, float]] = {
    "hazaribagh_town": HAZARIBAGH_TOWN,
    "hazaribagh_greater": HAZARIBAGH_GREATER,
    # full city uses a place-name query, not a bbox; bbox here is a fallback
    "hazaribagh_full": (24.0600, 23.9300, 85.4200, 85.3000),
}

DEFAULT_AREA = "hazaribagh_town"

# Full administrative city boundary (pulled by name from OSM on the user's
# machine). Use this for the complete Hazaribagh network rather than a tile.
CITY_PLACE_QUERY = "Hazaribagh, Jharkhand, India"

# Human-readable context strings used in docs / dashboard narrative
AREA_LABELS: Dict[str, str] = {
    "hazaribagh_town": "Hazaribagh town center (NH-33 / Matwari belt)",
    "hazaribagh_greater": "Greater Hazaribagh (town + Lake + Korra belt)",
    "hazaribagh_full": "Full Hazaribagh city (OSM administrative boundary)",
}


@dataclass
class SegmentationConfig:
    """Parameters for mask extraction / occlusion handling."""
    # Classical fallback threshold for grayscale road extraction
    bin_threshold: int = 127
    # Morphological close kernel (px) to seal small gaps before skeletonizing
    close_kernel: int = 5
    # Remove connected components smaller than this many px (noise)
    min_component_px: int = 60
    # Synthetic occlusion simulation (for the "see-through" demo)
    occlusion_count: int = 40
    occlusion_max_radius: int = 16
    occlusion_seed: int = 7


@dataclass
class HealConfig:
    """Topological healing (MST + Disjoint Set gap bridging)."""
    # Maximum Euclidean gap (px) we are willing to bridge
    max_gap_px: float = 45.0
    # Maximum angular deviation (degrees) between the two stub directions
    # for a bridge to be considered "natural"
    max_angle_deg: float = 35.0
    # Weight blends distance and angle: cost = dist * (1 + angle_penalty*angle_norm)
    angle_penalty: float = 1.5


@dataclass
class CriticalityConfig:
    """Centrality + ablation parameters."""
    # k for approximate betweenness (None = exact). Use int on large graphs.
    betweenness_k: int = None
    # How many top gatekeeper nodes to ablate in the stress test
    ablation_top_n: int = 8
    # Number of random source/target pairs for path-length sampling
    path_sample_pairs: int = 250
    sample_seed: int = 11


@dataclass
class PipelineConfig:
    mode: str = "image"               # image | osm | synth | satellite
    area: str = DEFAULT_AREA          # used by osm mode
    out_dir: str = "outputs"
    unet_weights: str = "outputs/unet.pt"  # for satellite mode DL segmentation
    seg: SegmentationConfig = field(default_factory=SegmentationConfig)
    heal: HealConfig = field(default_factory=HealConfig)
    crit: CriticalityConfig = field(default_factory=CriticalityConfig)
    simulate_occlusion: bool = True   # apply synthetic occlusions in image mode
    verbose: bool = True


def get_area_bbox(area: str) -> Tuple[float, float, float, float]:
    if area not in AREAS:
        raise ValueError(
            f"Unknown area '{area}'. Choose from: {list(AREAS)}"
        )
    return AREAS[area]
