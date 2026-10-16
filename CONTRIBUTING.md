# Contributing

Create a repository-local virtual environment and install `.[render,mujoco,dev]` as
shown in the README. Keep caches, temporary files and experiment outputs inside the
checkout when working under a project-directory boundary.

Research changes must preserve the implemented EQN operators, loss functions,
probability/reward conventions, DAC replay masks and update order. Validate changed
mathematics and collection boundaries with direct numerical checks and short
training runs. An execution repair that changes observable behavior
must be recorded in `docs/modernization.md`.

Run CLI smoke checks, the validation pipeline and a package build before proposing
changes. Unit-test suites were removed at the project owner's request. Report missing optional runtimes and
unverified experiment behavior accurately. Do not remove a historical variant
merely because no current import references it: it may encode a research experiment.
