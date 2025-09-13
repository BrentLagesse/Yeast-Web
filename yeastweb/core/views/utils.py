import cv2
import hashlib
import json
from pathlib import Path
from yeastweb.settings import MEDIA_ROOT

# base_path = "data/images/"
# new_path = "data/ims/"
# for infile in os.listdir(base_path):
#     print ("file : " + infile)
#     read = cv2.imread(base_path + infile)
#     outfile = infile.split('.')[0] + '.jpg'
#     cv2.imwrite(new_path+outfile,read,[int(cv2.IMWRITE_JPEG_QUALITY), 200])


def tif_to_jpg(tif_path :Path, output_dir :Path) -> Path:
    filename = tif_path.stem
    read = cv2.imread(str(tif_path))
    temp =filename+ '.jpg'
    jpg_path = Path(output_dir / temp)
    cv2.imwrite(str(jpg_path), read,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return jpg_path


# Progress helpers (shared across views)
def progress_path(key: str) -> Path:
    p = Path(MEDIA_ROOT) / 'progress'
    p.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(key.encode('utf-8')).hexdigest()
    return p / f"{digest}.json"


def write_progress(key: str, phase: str) -> None:
    try:
        progress_path(key).write_text(json.dumps({"phase": phase}))
    except (OSError, IOError, PermissionError):
        # Best-effort only; never break main processing on failure
        pass
