import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def shards(tmp_path_factory):
    from swarm.data.prepare import prepare

    d = tmp_path_factory.mktemp("shards")
    prepare(d, dataset="synthetic", n_samples=4000, nodes=12, alpha=0.3, seed=5, min_size=40)
    return d
