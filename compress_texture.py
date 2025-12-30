import subprocess

FFMPEG = "ffmpeg"

# -------- PATHS --------
INPUT_PATTERN = "/media/frozzzen/DataDrive/ChromeDownloads/Dancer_dataset/C4/dancer_fr%04d.png"
OUTPUT_HEVC = "/media/frozzzen/DataDrive/ChromeDownloads/Dancer_dataset/compressed/texture/dancer.hevc"
OUTPUT_PNG_PATTERN = "/media/frozzzen/DataDrive/ChromeDownloads/Dancer_dataset/compressed/texture/dancer_fr%04d.png"

# -------- BASIC SETTINGS, you don't need to change these --------
FPS = 30
PRESET = "medium"
PIX_FMT = "yuv444p"

# -------- CHOOSE ONE MODE, you can vary CRF to get different quality, or simply set target bitrate --------
CRF = 1        # set to None to disable CRF mode (0-51, 0,1 -> lossless, 51 -> worst quality)
BITRATE = None # e.g. "5M" (set CRF=None if using this)

# -------------------------------

assert (CRF is None) ^ (BITRATE is None), "Choose either CRF or BITRATE, not both."

# Build encode command
encode_cmd = [
    FFMPEG,
    "-y",
    "-framerate", str(FPS),
    "-i", INPUT_PATTERN,
    "-c:v", "libx265",
    "-preset", PRESET,
    "-pix_fmt", PIX_FMT,
]

if CRF is not None:
    encode_cmd += ["-crf", str(CRF)]
else:
    encode_cmd += ["-b:v", BITRATE]

encode_cmd.append(OUTPUT_HEVC)

# Decode command
decode_cmd = [
    FFMPEG,
    "-y",
    "-i", OUTPUT_HEVC,
    OUTPUT_PNG_PATTERN
]

# Run
print("Encoding...")
subprocess.run(encode_cmd, check=True)

print("Decoding...")
subprocess.run(decode_cmd, check=True)

print("Done.")
