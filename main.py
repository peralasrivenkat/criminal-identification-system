from __future__ import annotations

import argparse

from database.db_operations import initialize_database, seed_database_from_dataset
from src.pipeline import identify_from_video, predict_image, train_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Criminal Identification System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("gui", help="Open the desktop home page")
    subparsers.add_parser("init-db", help="Initialize the local database schema")
    subparsers.add_parser("seed-db", help="Seed the database using the dataset folder")
    subparsers.add_parser("train", help="Train the PCA + ACO + MLP pipeline")

    identify_parser = subparsers.add_parser("identify", help="Predict the criminal ID from an image")
    identify_parser.add_argument("image_path", help="Path to the image file")

    video_parser = subparsers.add_parser("identify-video", help="Predict the criminal ID from a video")
    video_parser.add_argument("video_path", help="Path to the video file")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "gui":
        from gui.home import launch_home

        launch_home()
    elif args.command == "init-db":
        initialize_database()
        print("Database initialized successfully.")
    elif args.command == "seed-db":
        total = seed_database_from_dataset()
        print(f"Seeded {total} criminal record(s) from the dataset.")
    elif args.command == "train":
        metrics = train_pipeline()
        print(f"Training completed: {metrics}")
    elif args.command == "identify":
        result = predict_image(args.image_path)
        print(result)
    elif args.command == "identify-video":
        result = identify_from_video(args.video_path)
        print(result)


if __name__ == "__main__":
    main()
