import os
import subprocess
import sys


def main():
    # Get the virtual environment name, this is the full path
    venv_path = os.getenv('VIRTUAL_ENV')

    if venv_path is None:
        print("No virtual environment found. Please activate your virtual environment.")
        sys.exit(1)

    venv_name = os.path.basename(venv_path)  # extract the basename of the virtual environment

    # Construct the pip command, syntax of the command: python -m ipykernel install --user --name=venv-name
    command = [sys.executable, '-m', 'ipykernel', 'install', '--user', '--name={}'.format(venv_name)]

    # Run the pip command
    try:
        subprocess.run(command, check=True)
        print(f"IPython kernel installed for virtual environment: {venv_name}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while installing the IPython kernel: {e}")


if __name__ == "__main__":
    main()
