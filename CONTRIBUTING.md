# Contributing

Create a repository-local virtual environment and install `.[test,mujoco,dev]` as
shown in the README. Keep caches, temporary files and experiment outputs inside the
checkout when working under a project-directory boundary.

Research changes must preserve the implemented EQN operators, loss functions,
probability/reward conventions, DAC replay masks and update order. Add analytic or
finite-difference tests for changed mathematics and boundary tests for collection
or demonstration changes. An execution repair that changes observable behavior
must be recorded in `docs/modernization.md`.

Run the core research suite, applicable upstream tests, CLI smoke checks and a
package build before proposing changes. Report missing optional runtimes and
unverified experiment behavior accurately. Do not remove a historical variant
merely because no current import references it: it may encode a research experiment.
