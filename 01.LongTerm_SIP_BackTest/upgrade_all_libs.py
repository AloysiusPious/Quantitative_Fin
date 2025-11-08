import subprocess
import sys


def upgrade_libraries():
    try:
        # Run pip list command to get a list of outdated packages
        result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--outdated', '--format=freeze'],
                                capture_output=True, text=True)
        outdated_packages = result.stdout.strip().split('\n')

        if not outdated_packages:
            print("All packages are up to date.")
            return

        # Extract package names from the output and format them for pip upgrade command
        packages_to_upgrade = [package.split('==')[0] for package in outdated_packages if package.strip()]

        if not packages_to_upgrade:
            print("No valid packages to upgrade.")
            return

        # Run pip install command to upgrade the packages
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade'] + packages_to_upgrade, check=True)
        print("All packages upgraded successfully.")

    except subprocess.CalledProcessError as e:
        print("Error:", e)


if __name__ == "__main__":
    upgrade_libraries()
