"""Pre-clone all repos needed by swe_smith training data into a local bare cache.

Usage:
    python scripts/precache_repos.py --data ./data/swe_smith/train.parquet --cache /sharedata/repo_cache

Each repo is stored as a bare git repo at <cache>/<owner>__<repo>.git
Rollout workers can then clone from local cache instead of GitHub.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import polars as pl


def parse_repo_field(repo_field: str) -> str:
    """swesmith/owner__repo.commithash -> owner/repo"""
    name = repo_field.replace("swesmith/", "")
    parts = name.rsplit(".", 1)
    return parts[0].replace("__", "/")


def get_unique_repos(parquet_path: str) -> list[tuple[str, str]]:
    """Return sorted list of (github_owner/repo, base_commit)."""
    df = pl.read_parquet(parquet_path)
    pairs = set()
    for row in df.select("repo").unique().iter_rows():
        repo_field = row[0]  # swesmith/owner__repo.commithash
        name = repo_field.replace("swesmith/", "")
        # owner__repo.commithash -> (owner/repo, commithash)
        parts = name.rsplit(".", 1)
        github_repo = parts[0].replace("__", "/")
        commit = parts[1] if len(parts) > 1 else None
        pairs.add((github_repo, commit))
    return sorted(pairs)


def cache_key(github_repo: str) -> str:
    """owner/repo -> owner__repo.git"""
    return github_repo.replace("/", "__") + ".git"


def clone_bare(github_repo: str, cache_dir: Path, commit: str) -> bool:
    """Clone a bare repo into cache. Returns True if successful."""
    dest = cache_dir / cache_key(github_repo)
    if dest.exists():
        return True

    print(f"  Cloning {github_repo} (bare)...")
    try:
        subprocess.run(
            ["git", "clone", "--mirror",
             f"https://github.com/{github_repo}.git",
             str(dest)],
            check=True, capture_output=True, text=True, timeout=600
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        err = getattr(e, 'stderr', str(e))
        print(f"  FAILED: {github_repo}: {err[:200]}")
        # Clean up partial clone
        if dest.exists():
            subprocess.run(["rm", "-rf", str(dest)], capture_output=True)
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to train.parquet")
    parser.add_argument("--cache", required=True, help="Path to bare repo cache dir")
    parser.add_argument("--retry", type=int, default=3, help="Retries per repo")
    args = parser.parse_args()

    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    repos = get_unique_repos(args.data)
    print(f"Repos to cache: {len(repos)}")
    print(f"Cache dir: {cache_dir}")

    ok, fail = 0, 0
    for github_repo, commit in repos:
        success = False
        for attempt in range(args.retry):
            if clone_bare(github_repo, cache_dir, commit):
                success = True
                break
            if attempt < args.retry - 1:
                print(f"  Retrying {github_repo} ({attempt+2}/{args.retry})...")
        if success:
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} cached, {fail} failed out of {len(repos)}")
    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
