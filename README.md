# 4DVideoStreaming

## Get Rate-Distortion pairs

### Step 1: Compress texture using video codec
Install `ffmpeg` first.
Run `compress_texture.py`.
Note that 300 texture images will be looked as a video for compression, use the file size of `dancer.hevc` / 300 to calculate the size of per texture image and then calculate bitrate.

### Step 2: Compress mesh using Draco
Run `compression_draco.py`.

### Step 3: Evaluate
Run `evaluation.py`.
