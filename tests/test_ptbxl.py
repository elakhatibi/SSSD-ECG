import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from care_ecg.cli import _prepare_ptbxl
from care_ecg.data import (
    PTBXL_DIAGNOSTIC_SUPERCLASSES,
    ECGArrayBundle,
    iter_ptbxl,
    iter_ptbxl_diagnostic_superclasses,
    load_npz,
    load_ptbxl_diagnostic_aliases,
    load_ptbxl_diagnostic_mapping,
    map_scp_codes_to_diagnostic_superclasses,
    records_to_arrays,
)
from care_ecg.types import ECGRecord


class PTBXLDataTests(unittest.TestCase):
    def test_diagnostic_superclass_order(self):
        self.assertEqual(
            PTBXL_DIAGNOSTIC_SUPERCLASSES,
            ("NORM", "MI", "STTC", "CD", "HYP"),
        )

    def test_mapping_uses_maximum_likelihood_per_superclass(self):
        raw_codes = {"NORM": 65.0, "SR": 90.0, "IMI": 80.0, "OTHER": 100.0}
        code_to_superclass = {
            "NORM": "NORM",
            "SR": "NORM",
            "IMI": "MI",
            "OTHER": "RHYTHM",
        }

        mapped = map_scp_codes_to_diagnostic_superclasses(raw_codes, code_to_superclass)

        self.assertEqual(mapped, {"NORM": 90.0, "MI": 80.0})

    def test_mapping_loader_reads_diagnostic_rows_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scp_statements.csv"
            path.write_text(
                ",description,diagnostic,diagnostic_class\n"
                "NORM,normal ECG,1.0,NORM\n"
                "IMI,inferior myocardial infarction,1,MI\n"
                "SR,sinus rhythm,0,\n"
                "UNKNOWN,other diagnosis,true,OTHER\n",
                encoding="utf-8",
            )

            mapping = load_ptbxl_diagnostic_mapping(directory)

        self.assertEqual(mapping, {"NORM": "NORM", "IMI": "MI"})

    def test_alias_loader_groups_codes_and_descriptions_by_superclass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scp_statements.csv"
            path.write_text(
                ",description,diagnostic,diagnostic_class\n"
                "NORM,normal ECG,1,NORM\n"
                "IMI,inferior myocardial infarction,1,MI\n"
                "SR,sinus rhythm,0,\n",
                encoding="utf-8",
            )

            aliases = load_ptbxl_diagnostic_aliases(directory)

        self.assertIn("normal ECG", aliases["NORM"])
        self.assertIn("IMI", aliases["MI"])
        self.assertIn("inferior myocardial infarction", aliases["MI"])
        self.assertNotIn("sinus rhythm", aliases["NORM"])

    def test_iter_ptbxl_preserves_strat_fold_without_real_optional_dependencies(self):
        class Row:
            filename_hr = "records500/00000/00001_hr"
            filename_lr = "records100/00000/00001_lr"
            patient_id = 42
            scp_codes = "{'NORM': 100.0}"
            strat_fold = 9

            def __getitem__(self, name):
                return getattr(self, name)

        table = SimpleNamespace(iterrows=lambda: iter([(1, Row())]))
        fake_pandas = SimpleNamespace(read_csv=lambda *args, **kwargs: table)
        waveform = np.zeros((8, 12), dtype=np.float32)
        fake_wfdb = SimpleNamespace(rdsamp=lambda *args, **kwargs: (waveform, {}))

        with patch.dict(sys.modules, {"pandas": fake_pandas, "wfdb": fake_wfdb}):
            records = list(iter_ptbxl("/unused", sampling_rate=500))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].metadata["strat_fold"], 9)
        self.assertEqual(records[0].metadata["scp_codes"], {"NORM": 100.0})
        self.assertEqual(records[0].signal.shape, (12, 8))

    def test_diagnostic_iterator_maps_codes_and_keeps_raw_codes(self):
        source_records = [
            ECGRecord(
                signal=np.zeros((12, 4), dtype=np.float32),
                record_id="1",
                metadata={
                    "scp_codes": {"SR": 80.0, "NORM": 60.0},
                    "sampling_rate": 500,
                    "strat_fold": 10,
                },
            ),
            ECGRecord(
                signal=np.zeros((12, 4), dtype=np.float32),
                record_id="2",
                metadata={"scp_codes": {"SR": 90.0}, "strat_fold": 10},
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "scp_statements.csv").write_text(
                ",description,diagnostic,diagnostic_class\n"
                "NORM,normal ECG,1,NORM\n"
                "SR,sinus rhythm,0,\n",
                encoding="utf-8",
            )
            with patch("care_ecg.data.iter_ptbxl", return_value=iter(source_records)):
                records = list(iter_ptbxl_diagnostic_superclasses(directory))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].metadata["scp_codes"], {"NORM": 60.0})
        self.assertEqual(records[0].metadata["raw_scp_codes"], {"SR": 80.0, "NORM": 60.0})
        self.assertEqual(records[0].metadata["strat_fold"], 10)

    def test_records_to_arrays_preserves_folds_and_breaks_ties_by_label_order(self):
        records = [
            ECGRecord(
                signal=np.zeros((12, 4), dtype=np.float32),
                patient_id="p1",
                record_id="r1",
                metadata={
                    "scp_codes": {"NORM": 100.0, "MI": 100.0},
                    "sampling_rate": 1,
                    "strat_fold": 8,
                },
            ),
            ECGRecord(
                signal=np.ones((12, 4), dtype=np.float32),
                patient_id="p2",
                record_id="r2",
                metadata={
                    "scp_codes": {"NORM": 90.0},
                    "sampling_rate": 1,
                    "strat_fold": 10,
                },
            ),
        ]

        bundle = records_to_arrays(records, ("MI", "NORM"), target_rate=1, seconds=4)

        np.testing.assert_array_equal(bundle.labels, [0, 1])
        np.testing.assert_array_equal(bundle.folds, [8, 10])

    def test_load_npz_retains_optional_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.npz"
            np.savez(
                path,
                signals=np.zeros((2, 12, 4), dtype=np.float32),
                labels=np.asarray([0, 1]),
                folds=np.asarray([8, 10]),
            )

            bundle = load_npz(path, require_labels=True)

        np.testing.assert_array_equal(bundle.folds, [8, 10])

    def test_prepare_ptbxl_writes_official_splits_manifest_and_aliases(self):
        bundle = ECGArrayBundle(
            signals=np.zeros((4, 12, 4), dtype=np.float32),
            labels=np.asarray([0, 1, 2, 3]),
            folds=np.asarray([1, 8, 9, 10]),
            record_ids=np.asarray(["r1", "r2", "r3", "r4"]),
        )
        aliases = {name: [name] for name in PTBXL_DIAGNOSTIC_SUPERCLASSES}
        graph = SimpleNamespace(
            nodes=["dataset"],
            edges=["contains"],
            save=lambda path: Path(path).write_text("{}\n", encoding="utf-8"),
        )
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                root="/unused",
                output=directory,
                sampling_rate=500,
                seconds=10,
                min_likelihood=0.0,
                normalization="none",
            )
            with (
                patch("care_ecg.cli.records_to_arrays", return_value=bundle),
                patch("care_ecg.cli.iter_ptbxl_diagnostic_superclasses", return_value=iter(())),
                patch("care_ecg.cli.load_ptbxl_diagnostic_aliases", return_value=aliases),
                patch("care_ecg.cli.build_ptbxl_knowledge_graph", return_value=graph),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(_prepare_ptbxl(args), 0)

            train = load_npz(Path(directory) / "train.npz", require_labels=True)
            validation = load_npz(Path(directory) / "validation.npz", require_labels=True)
            test = load_npz(Path(directory) / "test.npz", require_labels=True)
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            saved_aliases = json.loads((Path(directory) / "scp_aliases.json").read_text())

        self.assertEqual(len(train.signals), 2)
        self.assertEqual(len(validation.signals), 1)
        self.assertEqual(len(test.signals), 1)
        self.assertEqual(manifest["records"], {"train": 2, "validation": 1, "test": 1})
        self.assertEqual(saved_aliases, aliases)


if __name__ == "__main__":
    unittest.main()
