"""VPG with the shared on-policy Gymnasium/MJWarp collector."""

from tf2rl.experiments.on_policy import main

if __name__ == "__main__":
    main(algorithm="vpg")
