"""Console script for fem_electrodiffusion_cellular_geometries."""

import argparse
def main():
    """Console script for fem_electrodiffusion_cellular_geometries."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('_', nargs='*')
    args = parser.parse_args()

    print("Arguments: " + str(args._))
    print("Replace this message by putting your code into "
          "fem_electrodiffusion_cellular_geometries.cli.main")
    return 0
