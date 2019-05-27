#!/bin/bash
TARGET_FILES=$(find /cs/engproj/322/raw_video/test_splitting -type f -name '*.mp4')
echo > duration.txt
for f in $TARGET_FILES
do
	echo "$f" >> duration.txt
	ffprobe -i $f -show_entries format=duration -sexagesimal -v quiet -of csv="p=0" >> duration.txt
done
