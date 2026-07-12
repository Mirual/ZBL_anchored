#!/bin/bash
# Build deepmd-kit (branch master = dev) from source with customized op (border_op) for PT.
# venv on top of conda-python deepmd (torch 2.8.0 is reused -> ABI matches
# libtorch, against which conda lmp is linked). DP_VARIANT=cpu: no nvcc,
# border_op is needed only as a comm primitive; the model math is torchscript CUDA.
set -euo pipefail
VENV=${ZBL_DP_DEVEL_VENV:-/path/to/dp-devel-venv}  # TODO: set for your machine
SRC=${ZBL_DEEPMD_SRC:-/path/to/lammps/deepmd-kit-src}  # TODO: set for your machine

${ZBL_PYTHON:-python} -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install --no-cache-dir -q cmake ninja scikit-build-core "setuptools-scm>=7"

# the deepmd-kit development branch is now master (the old devel is no longer on github);
# verified: master contains the DPA-3.1 flags (use_env_envelope etc.)
if [ ! -d "$SRC/.git" ]; then
  rm -rf "$SRC"
  git clone --depth 1 -b master https://github.com/deepmodeling/deepmd-kit.git "$SRC"
fi
cd "$SRC"
git rev-parse HEAD

export DP_ENABLE_PYTORCH=1
export DP_VARIANT=cpu
"$VENV/bin/pip" install --no-cache-dir --no-build-isolation -v . 2>&1 | tail -20

"$VENV/bin/python" - <<'PY'
import deepmd
print("deepmd devel:", deepmd.__version__)
import inspect
from deepmd.dpmodel.descriptor.dpa3 import RepFlowArgs
sig = sorted(inspect.signature(RepFlowArgs.__init__).parameters.keys())
need = ["use_env_envelope", "use_dynamic_sel", "use_new_sw", "use_torch_embed"]
print({k: (k in sig) for k in need})
from deepmd.pt.cxx_op import ENABLE_CUSTOMIZED_OP
print("ENABLE_CUSTOMIZED_OP:", ENABLE_CUSTOMIZED_OP)
PY
echo "BUILD_DONE"
