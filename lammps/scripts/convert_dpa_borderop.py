#!/usr/bin/env python3
"""Refreeze DPA-3.1 with border_op (for LAMMPS pair deepmd).

The source dpa-3.1-3m-ft.pth was frozen by a newer deepmd version without
customized op → LAMMPS crashes on border_op. Here: pt → .dp → pt in an env
where ENABLE_CUSTOMIZED_OP=True (conda deepmd 3.1.1, the same as lmp).

The only schema difference in the newer version is the kwarg `use_torch_embed`
in DescrptDPA3; in the model it equals False (the default) → it can be
safely dropped (checked by an assert).
"""
import sys

import deepmd.pt.model.descriptor.dpa3 as dpa3_mod

_orig_init = dpa3_mod.DescrptDPA3.__init__


def _patched_init(self, *args, **kwargs):
    flag = kwargs.pop("use_torch_embed", False)
    if flag:
        raise SystemExit("use_torch_embed=True: strip patch not applicable, stop.")
    return _orig_init(self, *args, **kwargs)


dpa3_mod.DescrptDPA3.__init__ = _patched_init

from deepmd.main import main  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
main(["convert-backend", src, dst])
print(f"OK: {src} -> {dst}")
