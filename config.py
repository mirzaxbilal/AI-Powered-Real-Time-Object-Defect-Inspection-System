MODEL_PATH = r"runs\classify\runs\classify\bottle_classifier-2\weights\best.pt"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.75
MIN_RECORD_INTERVAL = 2.0   # seconds between logged results (one bottle per interval)
BOX_SIZE = 300
MAX_HISTORY = 50
STREAM_FPS = 15

# Presence detection — static reference diff
# Frames captured at startup to build the "empty belt" reference image.
BG_WARMUP_FRAMES = 60
# Pixel-level difference (0-255) required to count a pixel as "changed".
# Lower = more sensitive; raise if belt texture causes false triggers.
DIFF_THRESHOLD = 25
# Fraction of ROI pixels that must differ from the reference before a bottle is declared present.
PRESENCE_THRESHOLD = 0.15
# How much the reference slowly adapts after each bottle clears (0 = never adapt, 1 = instant).
BG_ADAPT_RATE = 0.05

# AI analysis
FRAMES_DIR = "saved_frames"          # where rejected bottle frames are saved
LLM_MODEL  = "claude-opus-4-7"      # Claude model used for bottle analysis
