#!/usr/bin/env bash
#
# FFIAM one-click quickstart.  Clone the repo, then:
#
#     ./quickstart.sh                 # run the default NSTTF demo
#     ./quickstart.sh radial_small    # pick a different demo
#
# It builds the Docker image (first run only; cached afterwards), runs a demo
# analysis, and writes the plots and Excel summary to ./output on your machine.
# An NVIDIA GPU + Container Toolkit is used automatically when available;
# otherwise it falls back to the (slower) CPU backend so it still "just works".
#
# Available demos: nsttf (default), radial_small, sample_v2, radial
#   - radial is the full-size field (1.6 km, 10,000 heliostats) and needs a GPU.
#
# Requirements: Docker. For GPU runs: an NVIDIA GPU, recent driver, and the
# NVIDIA Container Toolkit (https://docs.nvidia.com/datacenter/cloud-native/).

set -euo pipefail
cd "$(dirname "$0")"

IMAGE="ffiam"
DEMO="${1:-nsttf}"
OUTDIR="$(pwd)/output"

# --- validate demo name --------------------------------------------------
case "$DEMO" in
  nsttf|radial_small|sample_v2|radial) ;;
  *) echo "ERROR: unknown demo '$DEMO'. Choose: nsttf, radial_small, sample_v2, radial" >&2; exit 1 ;;
esac

# --- prerequisites -------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: Docker is required. Install it: https://docs.docker.com/get-docker/" >&2
  exit 1
fi

mkdir -p "$OUTDIR"

# --- build the image (cached after the first run) ------------------------
echo ">> Building the FFIAM image (first run can take a few minutes)..."
docker build -t "$IMAGE" .

# --- pick a backend ------------------------------------------------------
# Probe whether '--gpus all' actually exposes a GPU inside the container.
GPU_RUN_ARGS=()
ENV_ARGS=(-e XDG_DATA_HOME=/opt/ffiam/output)
if docker run --rm --gpus all --entrypoint nvidia-smi "$IMAGE" -L >/dev/null 2>&1; then
  echo ">> NVIDIA GPU detected — using the CUDA backend."
  GPU_RUN_ARGS=(--gpus all)
else
  echo ">> No usable GPU / NVIDIA Container Toolkit — using the CPU backend (slower)."
  if [ "$DEMO" = "radial" ]; then
    echo "   NOTE: the full-size 'radial' demo needs a GPU; try 'radial_small' on CPU." >&2
  fi
  # cudart is present in the image even without --gpus, so force CPU explicitly.
  ENV_ARGS+=(-e FFIAM_FORCE_CPU=1)
fi

# --- run the demo --------------------------------------------------------
echo ">> Running demo: ${DEMO}"
docker run --rm "${GPU_RUN_ARGS[@]}" "${ENV_ARGS[@]}" \
  -v "$OUTDIR:/opt/ffiam/output" \
  "$IMAGE" -c "from pyffiam.examples import assess_${DEMO}; assess_${DEMO}()"

echo ""
echo ">> Done. Results are in: ${OUTDIR}/FFIAM/"
echo "   (Excel summary + PNG plots for the '${DEMO}' analysis.)"
