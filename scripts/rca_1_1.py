#!/usr/bin/env python3

from pathlib import Path

import unidiff
import yaml

DISTANCE_BOUND = 65

VULN_DIR = Path(__file__).parent.parent / "vuln"
for vuln_dir in VULN_DIR.iterdir():
    config_yaml = vuln_dir / "config.yaml"
    if config_yaml.exists():
        config_data = yaml.safe_load(config_yaml.read_text())
        hunk_count = config_data["hunk_count"]
        covered_count = config_data["covered_count"]
        year = int(config_data["datetime"][:4])

        if hunk_count == 1 and covered_count == 1 and year >= 2023:
            patch_diff = vuln_dir / "patch.diff"
            unidiff_patch = unidiff.PatchSet(patch_diff.read_text().splitlines())

            files = [file for file in unidiff_patch]
            assert len(files) == 1
            file = files[0].path

            hunks = [hunk for hunk in files[0]]
            assert len(hunks) == 1
            hunk = hunks[0]

            start_line = hunk.target_start
            end_line = hunk.target_start + hunk.target_length - 1

            middle_line = (start_line + end_line) // 2
            report = (vuln_dir / "report.txt").read_text()

            rc_distance = None
            approx_root_cause = f"{file}:{middle_line}"
            for distance in range(DISTANCE_BOUND):
                lower_bound = middle_line - distance
                upper_bound = middle_line + distance

                if f"{file}:{lower_bound}" in report:
                    rc_distance = distance
                    break

                if f"{file}:{upper_bound}" in report:
                    rc_distance = distance
                    break

            assert rc_distance is not None
            print(f"{vuln_dir.name}\t{rc_distance}\t{approx_root_cause}")
