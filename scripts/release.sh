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
VERSION_FILE="src/twod_mcda/__init__.py"
CHANGELOG_FILE="CHANGELOG.md"

cd "${PROJECT_ROOT}"

[[ ${VERSION} =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
    || fail "version must use the MAJOR.MINOR.PATCH format."

CURRENT_BRANCH=$(git branch --show-current)
[[ ${CURRENT_BRANCH} == "develop" ]] \
    || fail "release preparation must start on develop (currently on ${CURRENT_BRANCH:-detached HEAD})."

git show-ref --verify --quiet refs/heads/main \
    || fail "local branch main does not exist."

if git show-ref --verify --quiet "refs/tags/${TAG}"; then
    fail "tag ${TAG} already exists."
fi

CODE_VERSION=$(python -c \
    'import ast, pathlib, sys; tree = ast.parse(pathlib.Path(sys.argv[1]).read_text()); print(next(node.value.value for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)))' \
    "${VERSION_FILE}") \
    || fail "could not read __version__ from ${VERSION_FILE}."

[[ ${CODE_VERSION} == "${VERSION}" ]] \
    || fail "requested version ${VERSION} does not match __version__ (${CODE_VERSION})."

OTHER_TRACKED_CHANGES=$(git diff --name-only HEAD -- . \
    | grep -Fxv "${VERSION_FILE}" \
    | grep -Fxv "${CHANGELOG_FILE}" \
    || true)
[[ -z ${OTHER_TRACKED_CHANGES} ]] \
    || fail "tracked files other than ${VERSION_FILE} and ${CHANGELOG_FILE} have uncommitted changes: ${OTHER_TRACKED_CHANGES//$'\n'/, }."

UNTRACKED_FILES=$(git ls-files --others --exclude-standard)
[[ -z ${UNTRACKED_FILES} ]] \
    || fail "untracked files are present: ${UNTRACKED_FILES//$'\n'/, }."

git commit -m "Prepare release ${VERSION}" -- "${VERSION_FILE}" "${CHANGELOG_FILE}"
git switch main
git merge --no-ff develop -m "Merge develop into main for ${TAG} release"

git tag -a "${TAG}" -m "Release ${VERSION}"
git switch develop
git merge --ff-only main

echo
echo "Release ${VERSION} prepared locally."
echo "Inspect the history, then publish it with: scripts/publish-release.sh ${VERSION}"
