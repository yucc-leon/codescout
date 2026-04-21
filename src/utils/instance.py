import argparse
import os
import subprocess
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# Local bare repo cache dir. Set REPO_CACHE env var or use default.
# Pre-populate with: python scripts/precache_repos.py --data ... --cache ...


def _get_repo_cache() -> str:
    return os.environ.get("REPO_CACHE", "")


def _cache_key(repo_name: str) -> str:
    """Convert repo_name to bare cache directory name.
    Handles both formats:
      - GitHub format: 'owner/repo' -> 'owner__repo.git'
      - SWE-smith format: 'swesmith/owner__repo.hash' -> 'owner__repo.git'
    """
    name = repo_name
    # Strip swesmith/ prefix
    if name.startswith("swesmith/"):
        name = name[len("swesmith/"):]
    # Strip .commithash suffix (8 hex chars after last dot)
    parts = name.rsplit(".", 1)
    if len(parts) == 2 and len(parts[1]) >= 7 and all(c in "0123456789abcdef" for c in parts[1]):
        name = parts[0]
    # owner/repo -> owner__repo (if slash present)
    name = name.replace("/", "__")
    return name + ".git"


def _clone_from_cache(repo_name: str, dest: str) -> bool:
    """Try cloning from local bare repo cache. Returns True if successful."""
    repo_cache = _get_repo_cache()
    if not repo_cache:
        return False
    cached = os.path.join(repo_cache, _cache_key(repo_name))
    if not os.path.isdir(cached):
        return False
    try:
        subprocess.run(
            ["git", "clone", "--no-hardlinks", cached, dest],
            check=True, capture_output=True, text=True, timeout=60,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def clone_instance(
    repo_name: str, commit_id: str, instance_id: str, output_dir: Path, patch: str | None = None
) -> bool:
    """
    Clone a repository at a specific commit into a separate directory.
    Uses local bare repo cache (REPO_CACHE env var) when available,
    falls back to GitHub clone.

    Args:
        repo_name: Repository name in format 'owner/repo'
        commit_id: Commit hash to checkout
        instance_id: Instance ID for directory naming
        output_dir: Base output directory

    Returns:
        (True, instance_path) if successful, (False, None) otherwise
    """
    instance_dir_name = f"{repo_name.replace('/', '_')}_{instance_id}"
    instance_path = output_dir / instance_dir_name

    if instance_path.exists():
        print(f"  ✓ Instance {instance_id} already exists")
        return True, instance_path

    try:
        # Try local cache first (fast, no network)
        if not _clone_from_cache(repo_name, str(instance_path)):
            # Fallback to GitHub
            subprocess.run(
                ["git", "clone",
                 f"https://github.com/{repo_name}.git",
                 str(instance_path)],
                check=True, capture_output=True, text=True, timeout=300,
            )

        # Extract commit hash from repo_name if not provided
        # SWE-smith format: swesmith/owner__repo.COMMITHASH
        if commit_id is None:
            parts = repo_name.rsplit(".", 1)
            if len(parts) == 2 and len(parts[1]) >= 7 and all(c in "0123456789abcdef" for c in parts[1]):
                commit_id = parts[1]

        if commit_id is not None:
            subprocess.run(
                ["git", "-C", str(instance_path), "checkout", commit_id],
                check=True, capture_output=True, text=True,
            )
            print(f"  ✓ Cloned {instance_id} at commit {commit_id[:8]}")

        if patch is not None:
            subprocess.run(
                ["git", "-C", str(instance_path), "apply"],
                input=patch,
                check=True, capture_output=True, text=True,
            )

        return True, instance_path
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        err = getattr(e, 'stderr', str(e))
        print(f"  ✗ Error cloning {instance_id}: {err[:300]}")
        # Clean up partial clone
        if instance_path.exists():
            subprocess.run(["rm", "-rf", str(instance_path)], capture_output=True)
        return False, None
