import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute candidate features + xmatch")

    parser.add_argument(
        "--path_field", type=str, default="../S82sub8_59.12", help="Path to field",
    )
    args = parser.parse_args()

    # create symbolic link to data
    os.symlink(args.path_field, "static")

    # create annotated folder
    os.makedirs("annotated", exist_ok=True)
