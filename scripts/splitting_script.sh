#!/bin/bash
ffmpeg -r 29.916666667 -i /cs/engproj/322/raw_video/video_all.mp4 -c:v libx264 -acodec copy -f segment -segment_time 60 -force_key_frames "expr:gte(t,n_forced*60)" -reset_timestamps 1 -map 0 /cs/engproj/322/raw_video/test_splitting/part%d.mp4
