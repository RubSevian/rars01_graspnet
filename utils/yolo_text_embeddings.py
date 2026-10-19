"""Startup-only CPU text encoding; no camera, ROS or actuator interfaces."""

import json
from pathlib import Path
import sys


def main():
    import numpy as np
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(1)
    model = YOLO(sys.argv[1])
    classes = json.loads(sys.stdin.read())
    with torch.inference_mode():
        embeddings = model.get_text_pe(classes).detach().cpu().numpy()
    np.save(Path(sys.argv[2]), embeddings, allow_pickle=False)


if __name__ == "__main__":
    main()
