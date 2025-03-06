import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--directory", type=str, help="Directory path.", required=True)

args = parser.parse_args()


videos = []
if os.path.isdir(args.directory):
    videos = [os.path.join(args.directory, file) for file in os.listdir(args.directory) if file.lower().endswith((".mp4", ".wav"))]

dbutils.jobs.taskValues.set(key="videos", value=videos)
