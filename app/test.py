from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import base64

# import numpy as np

# ---- Import your project code ----
# You’ll need to put your existing function into app/foot_size.py
from foot_size_estm import get_foot_size_estm


class MeasureResponse(BaseModel):
    length_mm: float
    width_mm: float
    image_vis_b64: Optional[str] = None


def measure():
    tmp_path = "../foot_size_estm/IMG_7350.jpg"
    # tmp_path = "../foot_size_estm/IMG_7349.jpg"
    gender = "m"
    ref_obj = "paper_letter"
    is_wall = True

    length_mm, width_mm, _ = get_foot_size_estm(
        tmp_path, gender=gender, ref_obj=ref_obj, is_wall=is_wall
    )

    resp = {
        "length_mm": float(length_mm),
        "width_mm": float(width_mm),
        "image_vis_b64": None,
    }

    print(resp)


measure()
