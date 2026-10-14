FROM python:3.11-slim
WORKDIR /project
COPY pyproject.toml setup.py README.md LICENSE ./
COPY tf2rl ./tf2rl
RUN python -m pip install --no-cache-dir '.[mujoco]'
CMD ["eqn-train", "--help"]
