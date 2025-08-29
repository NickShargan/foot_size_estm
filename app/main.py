from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import os
import base64

# import numpy as np

# ---- Import your project code ----
# You’ll need to put your existing function into app/foot_size.py
from .foot_size_estm import get_foot_size_estm


app = FastAPI(
    title="Foot Size Estimator",
    description="Estimate foot length/width (mm) and standardized shoe sizes\
                 from a single image.",
    version="1.0.0",
)


class MeasureResponse(BaseModel):
    length_mm: float
    width_mm: float
    image_vis_b64: Optional[str] = None


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/measure", response_model=MeasureResponse)
async def measure(
    file: UploadFile = File(..., description="Foot photo; formats: jpg/jpeg/png"),
    gender: str = Form("m"),
    ref_obj: str = Form("paper_letter"),
    is_wall: bool = Form(True),
    return_vis: bool = Form(False),
):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=415, detail="Unsupported file type")
    try:
        suffix = ".jpg" if "jpeg" in file.content_type or "jpg" in file.content_type else ".png"
        tmp_path = f"/tmp/upload{suffix}"
        with open(tmp_path, "wb") as f:
            f.write(await file.read())

        length_mm, width_mm = get_foot_size_estm(
            tmp_path, gender=gender, ref_obj=ref_obj, is_wall=is_wall
        )

        resp = {
            "length_mm": float(length_mm),
            "width_mm": float(width_mm),
            "image_vis_b64": None,
        }

        # if return_vis and vis_path and os.path.exists(vis_path):
        #     with open(vis_path, "rb") as vf:
        #         b64 = base64.b64encode(vf.read()).decode("utf-8")
        #         resp["image_vis_b64"] = f"data:image/jpeg;base64,{b64}"

        return JSONResponse(resp)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0",
                port=int(os.environ.get("PORT", 8080)),
                reload=False)
