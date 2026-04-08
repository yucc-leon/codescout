#!/bin/bash
# Git clone via GitHub API tarball download (bypasses git's libcurl-gnutls proxy issue)
# Usage: git_clone_via_api.sh <repo> <target_dir> [commit]
# Example: git_clone_via_api.sh pytest-dev/pytest /tmp/testbed/pytest abc123

set -euo pipefail

REPO="$1"
TARGET_DIR="$2"
COMMIT="${3:-}"

PROXY="http://user:passWorD@192.168.189.225:1080"

# Download tarball of default branch
TARBALL_URL="https://github.com/${REPO}/archive/refs/heads/main.tar.gz"
TMPTAR=$(mktemp /tmp/git_clone_XXXXXX.tar.gz)

echo "Downloading ${REPO}..."
curl -sL --proxy "$PROXY" -o "$TMPTAR" "$TARBALL_URL"

# Extract
mkdir -p "$TARGET_DIR"
tar xzf "$TMPTAR" --strip-components=1 -C "$TARGET_DIR"
rm -f "$TMPTAR"

# Init git repo so git commands work
cd "$TARGET_DIR"
git init -q
git add -A
git commit -q -m "initial" --allow-empty

# If commit specified, we need the full history - fetch via API
if [ -n "$COMMIT" ]; then
    echo "Fetching commit ${COMMIT}..."
    COMMIT_TAR="https://github.com/${REPO}/archive/${COMMIT}.tar.gz"
    TMPTAR2=$(mktemp /tmp/git_clone_XXXXXX.tar.gz)
    curl -sL --proxy "$PROXY" -o "$TMPTAR2" "$COMMIT_TAR"
    
    # Clean and re-extract at the specific commit
    git rm -rq --cached . 2>/dev/null || true
    rm -rf ./*
    tar xzf "$TMPTAR2" --strip-components=1 -C "$TARGET_DIR"
    rm -f "$TMPTAR2"
    
    git add -A
    git commit -q -m "at commit $COMMIT" --allow-empty
fi

echo "Done: $TARGET_DIR"
