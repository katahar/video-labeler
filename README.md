# Video Labeler

Desktop labeling tool for video gesture-classification datasets. It labels:

- object bounding boxes with stable track IDs,
- gesture frame ranges,
- configurable object and gesture vocabularies,
- optional SAM2 box-prompt propagation with manual fallback.

Annotations are saved beside each video as `<video_stem>.labels.json` in a COCO-style video JSON format.

## Setup

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

SAM2 is installed automatically into the active Python environment the first time
`P` is used. The app uses Meta's SAM 2.1 tiny pretrained model by default and
downloads its weights on first use. For GPU use, installing the matching PyTorch
build in advance is still recommended:

```bash
pip install torch torchvision
pip install git+https://github.com/facebookresearch/sam2.git
```

Custom model config and checkpoint paths can be set under `sam2.model_cfg` and
`sam2.checkpoint`. When both are empty, the automatic pretrained model is used.
If setup fails, the labeler remains usable in manual bounding-box mode.

Platform notes:

- macOS: the tool uses AppleScript for native pickers, then falls back to terminal path input. It avoids Tk because some macOS/Tk builds abort the Python process.
- Windows/Linux: the tool tries Tk native file/folder pickers, then falls back to terminal path input if Tk is unavailable.
- Linux: OpenCV GUI windows require a desktop session or X/Wayland forwarding.
- Windows: if PowerShell blocks virtualenv activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that shell and activate again.

## Label Config

Copy and edit the example:

```bash
cp labels.example.json labels.json
```

Example shape:

```json
{
  "object_categories": [
    {"id": 1, "name": "left_hand"},
    {"id": 2, "name": "right_hand"}
  ],
  "gesture_categories": [
    {"id": 1, "name": "reach"},
    {"id": 2, "name": "grasp"}
  ],
  "sam2": {
    "model_cfg": "",
    "checkpoint": "",
    "device": "auto"
  }
}
```

## Run

```bash
python video_labeler.py labels.json /path/to/videos
```

Windows example:

```powershell
py video_labeler.py labels.json C:\path\to\videos
```

Without arguments, the app opens native pickers for the input directory and label config.

If the selected directory contains an `mkv/` subfolder, that subfolder is used, matching the behavior of the reference `video_masker` tool.

## Controls

Menu:

- `<number>`: open a video
- `n`: open next unlabeled video
- `i`: change input directory
- `c`: change label config
- `r`: refresh
- `q`: quit

Editor:

- mouse drag: draw a box, or move an existing box
- right click: delete the box on the current frame
- scroll, arrows, `J`, `L`: scrub by one frame
- `A`, `D`: scrub by 10 frames
- `[`, `]`: scrub by 100 frames
- space: play/pause
- `1`..`9`: playback speed
- `N`: next drawn box starts a new object track
- `T`: cycle active object track
- `O`: cycle object category
- `G`: cycle gesture category
- `B`: begin gesture range at current frame
- `E`: end gesture range at current frame
- `X`: delete the gesture range containing the current frame
- `P`: initialize SAM2 from the active box and propagate it forward; propagated boxes remain draggable for correction, and `P` can be used again from a corrected box
- `S`: save annotations
- use the on-video dropdowns to select object types, gesture types, tracks, and existing gesture ranges
- scroll the mouse wheel while a dropdown is open to browse long lists of tracks or gestures
- after selecting an existing gesture, `B` and `E` move its start and end; selecting a gesture type relabels it
- `I`: save as in progress and return to the menu (the video remains eligible for `n`)
- `Q` or `Esc`: save as completed and return to the menu

## Output Format

Each video gets a sidecar file:

```text
example.mp4
example.labels.json
```

The JSON contains:

- `info.labeling_status`: `in_progress` or `completed`,
- `video`: video metadata,
- `images`: one COCO-style image record per frame,
- `categories`: object categories,
- `annotations`: per-frame bounding boxes with `track_id`,
- `tracks`: compact track-centric bounding-box records,
- `gesture_categories`: configured gesture labels,
- `gestures`: frame-range gesture annotations.

Bounding boxes use COCO `[x, y, width, height]` pixel coordinates. Gesture ranges use inclusive zero-based `start_frame` and `end_frame`.
