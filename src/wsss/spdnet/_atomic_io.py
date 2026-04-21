"""Atomic ``.npy`` I/O for probe seed generation.

Why this module exists
----------------------

``np.save(path, obj)`` for ``dtype=object`` arrays (which is what we use to
serialise per-class CAM dictionaries -- ``{0: float32_array}``) is **not
atomic** on POSIX filesystems. Internally NumPy:

  1. Opens ``path`` for write, truncating any existing file.
  2. Writes the ``\\x93NUMPY`` magic + header (~128 bytes).
  3. Streams ``pickle.dump(obj, fp)`` into the still-open handle.

If the process is killed, the host runs out of memory, or the kernel page
cache is evicted before fsync, the file on disk is the prefix of what was
intended -- typically rounded down to the nearest filesystem block (we
observed ``apple_mosaic_virus_google_0053.npy`` truncated to exactly
262 144 bytes / 256 KiB during the 18 Apr overnight run, mid-pickle
through a ~853 KiB write). On the next eval pass ``np.load`` then dies
with::

    _pickle.UnpicklingError: pickle data was truncated

which crashes the entire eval loop -- hard to recover from in an
unattended overnight run.

The fix is the standard "write-and-rename" idiom:

  1. Write to ``path.with_suffix(path.suffix + ".tmp")``.
  2. ``os.replace(tmp, dst)`` on success -- this is atomic on the same
     filesystem (``rename(2)`` semantics).
  3. On any exception, remove the ``.tmp`` so a partial write never
     survives as a "looks valid but is truncated" final file.

Combined with :func:`prune_corrupt_seeds` (which scans seed dirs for
already-truncated legacy files and deletes them so the seed-generator
loop refills them), this completely closes the silent-corruption hole
for the probe pipeline.

A 1-in-1200 corruption rate over Phase 1 alone (13 200 writes per phase)
is a near-certain crash; here we make the rate effectively zero and any
remaining corruption (e.g. cosmic ray on a previously-written file)
detectable + auto-recoverable.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def atomic_save_npy(dst: str | Path, obj: Any) -> None:
    """Write ``obj`` to ``dst`` atomically as a ``.npy`` file.

    Same call signature as ``np.save(str(dst), obj)``. Either the final
    file at ``dst`` is fully valid (passes ``np.load(...).item()``) or it
    is absent -- never half-written.

    Parameters
    ----------
    dst:
        Destination path. Must end in ``.npy`` (matches ``np.save``
        convention; a ``.npy`` is appended automatically by NumPy if
        omitted, but we want explicit control of the temp-file suffix).
    obj:
        Object to serialise. Same restrictions as ``np.save``.

    Notes
    -----
    The temp file is named ``<dst>.tmp``. We keep the ``.npy`` suffix on
    the temp file so that any future ``glob("*.npy")`` walker would
    *also* see the half-written file (rather than silently leaking) --
    this lets :func:`prune_corrupt_seeds` clean up after a hard kill,
    too.
    """
    dst = Path(dst)
    if dst.suffix != ".npy":
        raise ValueError(f"atomic_save_npy expects a .npy path, got {dst!r}")

    tmp = dst.with_name(dst.name + ".tmp")
    try:
        # IMPORTANT: ``np.save(str_path, ...)`` *appends* ``.npy`` to any
        # path that doesn't already end in ``.npy`` -- so a literal
        # ``np.save("x.npy.tmp", obj)`` would silently write to
        # ``x.npy.tmp.npy`` and leave our temp file (and the eventual
        # rename) pointing at nothing. Passing an open binary handle
        # bypasses that automatic-extension behavior.
        with open(tmp, "wb") as fh:
            np.save(fh, obj, allow_pickle=True)
        os.replace(tmp, dst)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def is_corrupt_npy(path: str | Path) -> bool:
    """Return True iff ``path`` cannot be loaded as a ``np.save`` payload.

    Covers the failure modes we have actually seen in production:

    * ``_pickle.UnpicklingError: pickle data was truncated``
      (mid-pickle truncation, typical 256 KiB / 4 KiB page boundary)
    * ``EOFError`` (zero-byte file from a ``open() + crash``)
    * ``ValueError: cannot reshape ...``
      (header OK but raw-array body short)
    * ``OSError: Failed to interpret file ...``
      (header itself is mangled)

    A ``False`` return guarantees the file passes ``.item()`` -- the
    same call eval makes, so any file we accept here will not crash
    downstream.
    """
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return True
    try:
        arr = np.load(str(p), allow_pickle=True)
        # Materialise the wrapped object exactly the way eval does --
        # otherwise a truncated *pickle* (vs a truncated *header*) slips
        # through the np.load call and only blows up later.
        if arr.dtype == object:
            arr.item()
    except (
        pickle.UnpicklingError,
        EOFError,
        ValueError,
        OSError,
    ):
        return True
    return False


def prune_corrupt_seeds(seed_dir: str | Path) -> list[Path]:
    """Delete every corrupt ``.npy`` (and stray ``.npy.tmp``) under ``seed_dir``.

    Returns the list of paths that were removed, sorted lexicographically.
    Idempotent: a clean directory yields ``[]``. Used at the start of
    :func:`scripts.eval_seg_probes.evaluate_probe` so legacy truncated
    files left over from before the atomic-write fix get rebuilt by the
    seed-generator on the next pass instead of crashing the loop.

    Notes
    -----
    * ``*.tmp`` files are *always* removed (they cannot be valid -- if
      the producer had finished, ``os.replace`` would have renamed them
      to ``.npy``).
    * Missing or empty directories are no-ops, not errors -- this lets
      the caller invoke us unconditionally on every seed dir without
      pre-checking existence.
    """
    sd = Path(seed_dir)
    removed: list[Path] = []
    if not sd.exists():
        return removed

    for stale in sorted(sd.glob("*.npy.tmp")):
        try:
            stale.unlink()
            removed.append(stale)
        except OSError:
            pass

    for f in sorted(sd.glob("*.npy")):
        if is_corrupt_npy(f):
            try:
                f.unlink()
                removed.append(f)
            except OSError:
                pass
    return removed


__all__ = ["atomic_save_npy", "is_corrupt_npy", "prune_corrupt_seeds"]
