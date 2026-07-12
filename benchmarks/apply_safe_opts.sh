#!/usr/bin/env bash
# ============================================================================
# apply_safe_opts.sh  —  applies the bit-identical safe fixes from the audit (B and D)
# to MACE 0.3.15 in the given env. Idempotent, with backups, reversible.
#
#   B: remove redundant .clone() in the energy upcast (models.py) — -2 allocations/forward
#   D: clamp_min(1e-6) on the Bessel 1/r denominator (radial.py) — guard against NaN forces
#      at r->0 (self-loops / tight PBC contacts). Below any physical distance
#      => no-op on real data, but removes 0/0 -> inf on the create_graph path.
#
# Both edits do NOT change the state_dict (weights load identically) and are bit-identical on
# physical inputs. Revert: restore the *.orig_safeopts files.
#
#   Run:      bash apply_safe_opts.sh [ /path/to/env ]   (default: mace-cueq)
#   Revert:   bash apply_safe_opts.sh [env] --revert
# ============================================================================
set -euo pipefail
ENV="${1:-/path/to/conda-env}"   # TODO: set for your machine (conda env with MACE 0.3.15)
PY="${ENV}/bin/python"
[ -x "${PY}" ] || { echo "ERROR: no python in ${ENV}"; exit 1; }
MACE_DIR="$("${PY}" -c 'import mace, os; print(os.path.dirname(mace.__file__))')"
RADIAL="${MACE_DIR}/modules/radial.py"
MODELS="${MACE_DIR}/modules/models.py"

if [ "${2:-}" = "--revert" ]; then
    for f in "${RADIAL}" "${MODELS}"; do
        if [ -f "${f}.orig_safeopts" ]; then mv -f "${f}.orig_safeopts" "${f}"; echo "reverted ${f}"; fi
    done
    "${PY}" -c 'import mace.modules.radial, mace.modules.models; print("import OK after revert")'
    exit 0
fi

"${PY}" - "${MACE_DIR}" <<'PYEOF'
import os, sys
d = sys.argv[1]
def patch(rel, old, new, tag):
    path = os.path.join(d, rel)
    s = open(path).read()
    if new in s:
        print(f"  {tag}: already applied"); return
    if old not in s:
        print(f"  {tag}: PATTERN NOT FOUND (mace version?) — skip"); return
    bak = path + ".orig_safeopts"
    if not os.path.exists(bak):
        open(bak, "w").write(s)
    open(path, "w").write(s.replace(old, new, 1))
    print(f"  {tag}: applied (backup {os.path.basename(bak)})")

# D — Bessel 1/r guard
patch("modules/radial.py",
      "return self.prefactor * (numerator / x)",
      "return self.prefactor * (numerator / x.clamp_min(1e-6))  # audit fix D: guard 1/r at r->0",
      "D (Bessel 1/r NaN-guard)")
# B — drop redundant clones in energy upcast
patch("modules/models.py",
      "node_e0.clone().double() + node_inter_es.clone().double()",
      "node_e0.double() + node_inter_es.double()  # audit fix B",
      "B (drop redundant clones)")
PYEOF

"${PY}" -c 'import mace.modules.radial, mace.modules.models; print("import OK after patch")'
echo "done. Revert: bash $0 ${ENV} --revert"
