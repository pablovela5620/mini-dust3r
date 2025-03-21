import tyro

from mini_dust3r.api.inference import (
    InferenceConfig,
    run_inference,
)

if __name__ == "__main__":
    run_inference(tyro.cli(InferenceConfig))
