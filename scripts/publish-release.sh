#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 VERSION" >&2
    echo "Example: $0 1.1.0" >&2
}

fail() {
    echo "Error: $*" >&2
    exit 1
}

if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi

VERSION=$1
TAG="v${VERSION}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null) \
    || fail "the script is not inside a Git repository."

cd "${PROJECT_ROOT}"

[[ ${VERSION} =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
    || fail "version must use the MAJOR.MINOR.PATCH format."

git diff --quiet && git diff --cached --quiet \
    || fail "the working tree contains tracked changes."

[[ -z $(git ls-files --others --exclude-standard) ]] \
    || fail "the working tree contains untracked files."

git show-ref --verify --quiet refs/heads/main \
    || fail "local branch main does not exist."
git show-ref --verify --quiet refs/heads/develop \
    || fail "local branch develop does not exist."
git show-ref --verify --quiet "refs/tags/${TAG}" \
    || fail "local tag ${TAG} does not exist."

TAG_COMMIT=$(git rev-list -n 1 "${TAG}")
MAIN_COMMIT=$(git rev-parse main)
DEVELOP_COMMIT=$(git rev-parse develop)

[[ ${TAG_COMMIT} == "${MAIN_COMMIT}" ]] \
    || fail "${TAG} does not point to the tip of main."
[[ ${DEVELOP_COMMIT} == "${MAIN_COMMIT}" ]] \
    || fail "develop is not aligned with main."

git remote get-url origin >/dev/null 2>&1 \
    || fail "remote origin is not configured."

echo "Publishing main and ${TAG} to origin atomically."
git push --atomic origin main "refs/tags/${TAG}"

echo "Release ${VERSION} published."
