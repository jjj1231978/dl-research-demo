#!/usr/bin/env bash
# Deploy the current main to the Hugging Face Space.
#
# The Space's history is intentionally squashed and separate from GitHub's.
# Hugging Face runs a pre-receive hook that rejects raw binaries anywhere in
# a pushed commit range, and this repo's history contains one (docs/hero.png
# was committed before *.png was tracked in LFS). Replaying full history
# would therefore be rejected, and rewriting it would mean force-pushing a
# repo that is already published on GitHub.
#
# So each deploy is one commit whose parent is whatever the Space is already
# at, carrying main's exact tree.
#
# Usage:  scripts/deploy_hf.sh ["commit message"]

set -euo pipefail
cd "$(dirname "$0")/.."

MSG="${1:-Deploy: $(git -C . log -1 --format=%s main)}"

git diff --quiet && git diff --cached --quiet || {
  echo "error: working tree is dirty; commit or stash first" >&2; exit 1
}

echo "==> fetching the Space's current head"
git fetch hf main --quiet
HF_HEAD=$(git rev-parse FETCH_HEAD)

echo "==> building a deploy commit on ${HF_HEAD:0:8} with main's tree"
git branch -f hf-deploy "$HF_HEAD"
git checkout --quiet hf-deploy
git checkout --quiet main -- .

# Files that live only in the working tree, never in a deploy.
git reset --quiet -- 01_statistics.ipynb START-HERE.md \
                     dl-research-demo-refinement.zip 2>/dev/null || true
git add -A
git reset --quiet -- 01_statistics.ipynb START-HERE.md \
                     dl-research-demo-refinement.zip 2>/dev/null || true

if git diff --cached --quiet; then
  echo "==> nothing to deploy; the Space already matches main"
  git checkout --quiet main
  exit 0
fi

git commit --quiet -m "$MSG"
echo "==> pushing to the Space"
git push hf hf-deploy:main
git checkout --quiet main
echo "==> done. Watch the build at:"
echo "    https://huggingface.co/spaces/JJ-JIN12345/dl-research-demo"
