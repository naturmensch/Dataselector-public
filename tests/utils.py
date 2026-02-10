import importlib.util
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd


def load_module_from_path(name: str, path: Path, register: bool = False):
    """
    Loads a module from a file path dynamically.
    If register=True, the module is temporarily added to sys.modules under `name`.
    Sets __spec__ and __package__ to help modules that rely on import context.
    """
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None:
        raise ImportError(f"Could not load spec for {name} from {path}")
    mod = importlib.util.module_from_spec(spec)

    # Provide minimal import metadata for modules that expect it
    mod.__spec__ = spec
    mod.__package__ = spec.name.rpartition(".")[0] if "." in spec.name else ""

    if register:
        sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        # If register was True, keep it; otherwise, avoid polluting sys.modules
        if not register and name in sys.modules:
            del sys.modules[name]
    return mod


class FakeFeatureExtractor:
    """Stub for src.feature_extractor.FeatureExtractor"""

    def __init__(self, *args, **kwargs):
        pass

    def extract_features_batch(self, image_paths, data_dir, batch_size=16):
        # Return zeros matching number of image_paths
        return np.zeros((len(image_paths), 16), dtype=np.float32)


class FakeMetadataProcessor:
    """Stub for src.metadata_processor.MetadataProcessor"""

    def __init__(self, csv_path):
        self.csv_path = csv_path

    def load_csv(self):
        return pd.read_csv(self.csv_path)

    def add_temporal_metadata(self):
        return self.load_csv()

    def resolve_image_paths(self, image_dir):
        df = self.load_csv()
        if "image_path" not in df.columns:
            df["image_path"] = pd.Series([None] * len(df))
        return df


def create_dummy_script(path: Path, marker: str = "DUMMY_RUN_DONE"):
    """Creates a standalone dummy python script for testing monitors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        textwrap.dedent(
            f"""
            #!/usr/bin/env python3
            import os, sys, time
            
            # Simulate work
            root = os.getcwd()
            outdir = os.path.join(root, 'outputs', 'runs', 'dummy_run')
            os.makedirs(outdir, exist_ok=True)
            
            # Write artifact
            with open(os.path.join(outdir, 'results.txt'), 'w') as f:
                f.write('dummy-result')
                
            print('{marker}')
            sys.stdout.flush()
            time.sleep(0.1)
            sys.exit(0)
            """
        )
    )
    path.chmod(0o755)
