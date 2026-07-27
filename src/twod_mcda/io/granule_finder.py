def find_neighbor_granules(
    granule_date: datetime,
    version_l1: str,
    type_l1: str,
    data_root: Path,
) -> tuple[Path | None, Path | None]:
    """
    Cherche les granules CALIOP L1 immédiatement avant et après
    `granule_date`, pour la version/type demandés.
    Retourne (None, None) si absentes (bord de mission, trou, etc.)
    """