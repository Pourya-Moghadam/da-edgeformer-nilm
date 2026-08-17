#!/usr/bin/env bash
set -euo pipefail

destination=${1:-external}
mkdir -p "$destination"

fetch_pinned() {
  local name=$1
  local repository=$2
  local commit=$3
  local checkout="$destination/$name"
  if [[ ! -d "$checkout/.git" ]]; then
    git clone "$repository" "$checkout"
  fi
  git -C "$checkout" fetch origin "$commit"
  git -C "$checkout" checkout --detach "$commit"
  actual=$(git -C "$checkout" rev-parse HEAD)
  if [[ "$actual" != "$commit" ]]; then
    echo "commit verification failed for $name" >&2
    exit 1
  fi
}

fetch_pinned \
  NILMFormer \
  https://github.com/adrienpetralia/NILMFormer.git \
  e73a975a42fbebed6f9d7e90d75f7f48ae02fed9
fetch_pinned \
  AugLPN_NILM \
  https://github.com/linfengYang/AugLPN_NILM.git \
  5e3b37c5ad91fa1a243d144d8ba133a78a6fb0d9
