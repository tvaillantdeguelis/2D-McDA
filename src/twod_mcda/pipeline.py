"""Top-level processing pipeline."""

from datetime import datetime
from pathlib import Path
import re
import time

from .io.granule_finder import find_granule_file, find_neighbor_granules
from .processing.granule_processor import process_granule as run_processing
from .processing.models import ProcessingRequest
from .version import get_full_version


_GRANULE_ID_PATTERN = re.compile(
    r"\.(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z[DN])\.hdf$"
)


def _granule_id(file_path):
    """Extract the full granule identifier, including day/night."""

    if file_path is None:
        return None

    file_path = Path(file_path)
    match = _GRANULE_ID_PATTERN.search(file_path.name)
    if match is None:
        raise ValueError(f"Invalid CALIOP filename format: {file_path.name}")

    return match.group(1)


def _normalized_version(version):
    """Return a version with the upper-case prefix used in product metadata."""

    if version[:1].lower() == "v":
        return f"V{version[1:]}"
    return f"V{version}"


def _altitude_index(max_altitude_km):
    """Map a supported maximum altitude to the regular-grid index."""

    if max_altitude_km == 30:
        return 600
    if max_altitude_km == 40:
        return None

    raise ValueError(
        "processing.max_altitude_km must be either 30 or 40."
    )


def _output_directory(output_cfg, granule_date, version):
    """Build the output directory from the configured root and path format."""

    relative_path = output_cfg["path_format"].format(
        version=version.removeprefix("V"),
        year=granule_date.year,
        month=granule_date.month,
        day=granule_date.day,
    )

    return Path(output_cfg["root_directory"]) / relative_path


def _processing_request(cfg, current_file, previous_file, next_file):
    """Build a processing request from the YAML configuration and inputs."""

    processing_cfg = cfg.get("processing", {})
    output_cfg = cfg["output"]
    subset_cfg = cfg.get("subset", {})
    caliop_cfg = cfg["cal_lid_l1"]
    version = _normalized_version(get_full_version())
    granule_date = _granule_id(current_file)
    granule_time = datetime.strptime(
        granule_date[:19],
        "%Y-%m-%dT%H-%M-%S",
    )

    return ProcessingRequest(
        granule_date=granule_date,
        caliop_version=_normalized_version(str(caliop_cfg["version"])),
        current_directory=Path(current_file).parent,
        previous_granule=_granule_id(previous_file),
        previous_directory=(
            Path(previous_file).parent if previous_file is not None else None
        ),
        next_granule=_granule_id(next_file),
        next_directory=(
            Path(next_file).parent if next_file is not None else None
        ),
        subset_mode=subset_cfg.get("mode", "profindex"),
        subset_start=subset_cfg.get("start"),
        subset_end=subset_cfg.get("end"),
        save_development_data=processing_cfg.get(
            "save_development_data", False
        ),
        output_version=version,
        output_product_type=output_cfg.get("product_type", "Dev"),
        output_directory=_output_directory(output_cfg, granule_time, version),
        maximum_altitude_km=processing_cfg["max_altitude_km"],
        maximum_altitude_index=_altitude_index(
            processing_cfg["max_altitude_km"]
        ),
    )


def process_granule(cfg):
    """Resolve granule paths and run the processing pipeline."""

    start_time = datetime.now().astimezone()
    start_tic = time.perf_counter()
    print(f"Start time: {start_time}\n")

    granule_time = datetime.strptime(cfg["granule"], "%Y-%m-%dT%H-%M-%SZN")
    current_file = find_granule_file(cfg, granule_time)
    previous_file, next_file = find_neighbor_granules(cfg, current_file)

    request = _processing_request(
        cfg,
        current_file,
        previous_file,
        next_file,
    )
    run_processing(request)

    end_time = datetime.now().astimezone()
    total_time = time.perf_counter() - start_tic
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"End time: {end_time}")
    print(f"Total runtime: {int(hours)} h {int(minutes)} min {seconds:.1f} s")
