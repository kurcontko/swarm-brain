#!/usr/bin/env bash
# Build the Swarm Brain container image locally.
#
# Local only. This never logs in to a registry, never pushes, and never touches
# an AWS account. Pushing to ECR is a separate, approval-gated step documented
# in deploy/README.md.
#
#   scripts/build_image.sh                  -> swarm-brain:dev
#   scripts/build_image.sh swarm-brain:abc1234
#   scripts/build_image.sh "swarm-brain:$(git rev-parse --short HEAD)"
#
# One image, two entry points. The tag you build here is the tag both ECS
# services run: the API uses the image's default CMD, the worker overrides it
# with `swarmbrain-worker`.
#
# Environment:
#   PLATFORM   target platform, e.g. linux/arm64 or linux/amd64.
#              Default: the host's. Set it explicitly when the build host and
#              the ECS task's cpuArchitecture differ — an architecture mismatch
#              fails at task start with `exec format error`, *after* the image
#              pull, so it reads as a runtime problem and is not one.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tag="${1:-swarm-brain:dev}"

if ! command -v docker >/dev/null 2>&1; then
  echo "build_image: docker is not installed" >&2
  exit 2
fi
if ! docker info >/dev/null 2>&1; then
  echo "build_image: the docker daemon is not reachable; start Docker and retry" >&2
  exit 2
fi

# `uv sync --frozen` in the Dockerfile treats the lock as the authority and
# fails the build without it. Catching that here names the cause instead of
# leaving a resolver error to be interpreted.
if [ ! -f "$repo_root/uv.lock" ]; then
  echo "build_image: uv.lock is missing; run 'uv lock' first" >&2
  exit 2
fi

build_args=(build --file "$repo_root/Dockerfile" --tag "$tag")
if [ -n "${PLATFORM:-}" ]; then
  build_args+=(--platform "$PLATFORM")
fi
build_args+=("$repo_root")

echo "build_image: docker ${build_args[*]}"
docker "${build_args[@]}"

echo
docker image ls "$tag" --format 'build_image: {{.Repository}}:{{.Tag}} {{.Size}}'
echo "build_image: api    -> docker run --rm -p 8080:8080 --env-file <env> $tag"
echo "build_image: worker -> docker run --rm --env-file <env> $tag swarmbrain-worker"
