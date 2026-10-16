# Historical Gym source

`gym/` is the original bundled Gym fork, retained with its license and custom assets
for provenance. It is not installed, imported or needed by modern experiments. The
runtime assets are packaged in `tf2rl/envs/assets`. Do not put this directory on
PYTHONPATH or install it for the modern training workflows.

Distinct old algorithm files and `examples/old` were moved to `python/` in the
2026-10-16 cleanup. See [the version map](python/README.md) and its source hashes.
The bundled Gym unit tests were removed at the project owner's request; the
environment source, assets and license remain for provenance.
The exact duplicate `tf2rl/algos/airl_old_v1.py` had no incoming references and was
removed; the original copy is also retained in the local working-tree backup.
