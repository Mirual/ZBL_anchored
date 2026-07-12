# patches/ — bit-identical safe optimisations for MACE

`mace_safe_opts.patch` collects **behaviour-preserving** performance edits to the
MACE source. Applied against **ACEsuit/mace commit `4d2da09`** (MACE v0.3.16-class,
after merge #1468). "Safe" means energies / forces / stress stay identical to the
unpatched model up to rounding — only wall-clock improves.

## Usage
```bash
# from your MACE source checkout at commit 4d2da09
git apply /path/to/patches/mace_safe_opts.patch
```

`apply_safe_opts.sh` is a convenience applier that patches an installed MACE in a
target environment. Note it applies a subset (hunks B + D) of the full patch
(which also carries hunk A); read the script header before use and prefer
`git apply` on a source checkout for the complete set.

The numerical-equivalence of these edits is validated by the scripts in
`../benchmarks/` (`validate_opt4.py`, `validate_036*.py`).

MACE itself is MIT-licensed; this patch is provided under the same terms. The MACE
source is **not** vendored here — obtain it from https://github.com/ACEsuit/mace.
