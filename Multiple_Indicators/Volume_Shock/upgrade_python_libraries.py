import subprocess

# Upgrade pip itself
subprocess.run(['pip', 'install', '--upgrade', 'pip'])

# List outdated packages
outdated_packages = subprocess.run(['pip', 'list', '--outdated'], capture_output=True, text=True)

# Extract package names from the output
package_names = [line.split()[0] for line in outdated_packages.stdout.split('\n')[2:-1]]

# Upgrade each package
for package in package_names:
    subprocess.run(['pip', 'install', '--upgrade', package])
