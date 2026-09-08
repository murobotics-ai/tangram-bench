"""Fetch only the robot assets we use, at a fixed upstream revision."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "menagerie"
REVISION = "8161bba264d7fa7c99ca301e91e7fb44737676ad"
ROBOTS = {"panda": "franka_emika_panda/panda.xml", "piper": "agilex_piper/piper.xml"}


def prepare():
    if ASSETS.exists():
        revision = subprocess.check_output(
            ["git", "-C", str(ASSETS), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != REVISION:
            raise RuntimeError(f"Asset revision mismatch: {revision}; expected {REVISION}.")
        for filename in ROBOTS.values():
            if not (ASSETS / filename).is_file():
                raise RuntimeError(
                    f"Missing asset {filename}; remove the incomplete assets/menagerie and rerun."
                )
        print(f"Robot assets ready: {REVISION}")
        return
    ASSETS.mkdir(parents=True)

    def git(*args):
        subprocess.run(["git", "-C", str(ASSETS), *args], check=True)

    git("init", "-q")
    git("remote", "add", "origin", "https://github.com/google-deepmind/mujoco_menagerie.git")
    git("sparse-checkout", "init", "--cone")
    git("sparse-checkout", "set", "franka_emika_panda", "agilex_piper")
    git("fetch", "--depth", "1", "origin", REVISION)
    git("checkout", "--detach", "FETCH_HEAD")
    print(f"Robot assets ready: {REVISION}")


if __name__ == "__main__":
    prepare()
