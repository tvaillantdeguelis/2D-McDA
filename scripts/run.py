cfg = load_config(...)

if cfg["run"]["mode"] == "granule":
    granules = [cfg["run"]["granule"]]

elif cfg["run"]["mode"] == "period":
    granules = find_granules(cfg)

for granule in granules:
    process_granule(granule, cfg)