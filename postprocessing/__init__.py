from argparse import ArgumentParser

from postprocessing.convert import convert
from postprocessing.plot import plot

def main():
    parser = ArgumentParser(description="Postprocessing")

    operations = ("convert", "plot")

    parser.add_argument(
        "operation",
        choices=operations,
        help="The operation to be performed.",
    )
    args, unknown_args = parser.parse_known_args()

    func = globals()[args.operation]
    func(unknown_args)

if __name__ == "__main__":
    main()