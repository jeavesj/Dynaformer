#! /bin/bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 PKL_PATH NAME_TAG"
  exit 1
fi

PKL_PATH="$1"
NAME_TAG="$2"

bash Dynaformer/examples/evaluate/evaluate.sh \
  "" \
  "custom:path=${PKL_PATH}" \
  "" \
  "${NAME_TAG}"
