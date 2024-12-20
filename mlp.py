import argparse
from src.predict import predict
from src.split import split
from src.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="mlp.py",
        description="[description]",
    )

    parser.add_argument("command")

    args = parser.parse_args()

    if args.command == "split":
        split()
    elif args.command == "train":
        train()
    elif args.command == "predict":
        predict()
    else:
        exit("Fatal: not recognized command")
