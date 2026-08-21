# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

import json
from pathlib import Path

from pyffiam import utils
from pyffiam.analysis import analysis


def run_from_config_file():
    """Run analysis for every JSON file in site_configs/loaded/."""
    script_dir = Path(__file__).parent
    loaded_dir = script_dir / "site_configs" / "loaded"

    if not loaded_dir.exists() or not loaded_dir.is_dir():
        print(f"Error: Loaded directory not found: {loaded_dir}")
        return

    config_files = list(loaded_dir.iterdir())

    if not config_files:
        print(f"No configuration files found in {loaded_dir}")
        return

    id = 0
    for config_filepath in config_files:
        if config_filepath.suffix.lower() != '.json':
            print(f"Skipping non-JSON file: {config_filepath.name}")
            continue

        print(f"\nProcessing: {config_filepath.name}")

        try:
            config_dict = utils.get_site_config_dict_from_json(config_filepath)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {config_filepath}: {e}")
            continue

        id += 1
        run_id = config_dict.pop("RunId", f"run_{id}")
        print(f"\n\n==================================================================")
        print(f"--- Running Analysis: {run_id} ---")

        try:
            print('Calling analysis function...')

            config_dict['open_output_dir'] = True
            result = analysis(**config_dict)

            if result and hasattr(result, 'total_irrad'):
                print(f"Result for {run_id}: Total irradiance: {result.total_irrad:,.0f}")
            else:
                print(f"Result for {run_id}: Analysis completed (no total_irrad attribute or result is None).")
            print(f"==================================================================\n")

        except Exception as e:
            print(f"Error running analysis for {run_id}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    print('Beginning analyses...')
    run_from_config_file()
