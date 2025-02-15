import json
from typing import Any

import docker
from docker.models.containers import Container

STANDY_CONTAINER_COUNT = 2

# Pre-create standby Docker containers
client = docker.from_env()
standby_containers = []

for _ in range(STANDY_CONTAINER_COUNT):
    container = client.containers.run(
        "python:3.9", command="sleep infinity", detach=True, tty=True
    )
    standby_containers.append(container.id)


def clean_container(container: Container, packages: list[str]):
    """Uninstalls only the packages that were installed."""
    if packages:
        uninstall_cmd = f"pip uninstall -y {' '.join(packages)}"
        container.exec_run(uninstall_cmd)
    container.exec_run("rm -rf /tmp/*")


def execute_python_with_packages_in_docker(
    packages: list[str], code: str
) -> dict[str, Any]:
    try:
        if not standby_containers:
            return {"error": "No standby containers available"}

        container_id = standby_containers.pop(0)
        container = client.containers.get(container_id)

        # Install required packages inside the container
        install_cmd = f"pip install {' '.join(packages)}"
        container.exec_run(install_cmd)

        # Execute the provided Python code inside the container
        result = container.exec_run(f'python -c "{code}"')

        # Clean up the container before returning to standby
        clean_container(container, packages)

        # Return the container to standby
        standby_containers.append(container_id)

        return {
            "stdout": result.output.decode("utf-8"),
            "stderr": "" if result.exit_code == 0 else "Error executing code",
            "returncode": result.exit_code,
        }
    except Exception as e:
        return {"error": str(e)}


# Example usage:
response = execute_python_with_packages_in_docker(
    ["numpy"], "import numpy as np; print(np.array([1, 2, 3]))"
)
print(json.dumps(response, indent=2))
