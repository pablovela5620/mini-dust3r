import subprocess
from pathlib import Path

PIXI_PATH = Path("/home/user/.pixi/bin/pixi")
PIXI_VERSION = "0.32.1"


def check_and_install_pixi() -> None:
    try:
        subprocess.check_call(f"{PIXI_PATH} --version", shell=True)
    except subprocess.CalledProcessError:
        print("pixi not found. Installing pixi...")
        # Install pixi using the provided installation script
        subprocess.check_call(
            f"PIXI_VERSION=v{PIXI_VERSION} curl -fsSL https://pixi.sh/install.sh | bash",
            shell=True,
        )
        subprocess.check_call(
            f"{PIXI_PATH} self-update --version {PIXI_VERSION}", shell=True
        )
        subprocess.check_call(f"{PIXI_PATH} --version", shell=True)


def run_command(command: str) -> None:
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"run command {command}. Error: {e}")


if __name__ == "__main__":
    check_and_install_pixi()
    # install lsof
    run_command(command=f"{PIXI_PATH} global install lsof")
    # kill anything running on port 7860
    run_command(command=f"{PIXI_PATH.parent}/lsof -t -i:7860 | xargs -r kill")
    # clean current environment
    run_command(command=f"{PIXI_PATH} clean")
    # run spaces app
    run_command(command=f"{PIXI_PATH} run -e spaces app")
