# ui.py
import os
import io
import base64
import mimetypes
from typing import Tuple, Optional, Union

import requests
from PIL import Image
import gradio as gr

from utils import is_a4_format


API_BASE = os.environ.get(
    "API_BASE",
    "https://foot-size-api-762504128529.northamerica-northeast1.run.app"
)
API_URL = f"{API_BASE}/measure"

AFFILIATE_URL = os.environ.get("AMAZON_AFFILIATE_URL", "https://amzn.to/3VpAZ3r")
IS_AFFILIATE = bool(AFFILIATE_URL and ("tag=" in AFFILIATE_URL or "amzn.to" in AFFILIATE_URL))


def _decode_b64_image(data_uri_or_b64: str) -> Optional[Image.Image]:
    """
    Accepts either a full data URI (e.g., 'data:image/jpeg;base64,...')
    or a bare base64 string. Returns a PIL.Image or None on failure.
    """
    try:
        if "," in data_uri_or_b64 and data_uri_or_b64.strip().lower().startswith("data:"):
            # data URI -> split header and payload
            b64_str = data_uri_or_b64.split(",", 1)[1]
        else:
            b64_str = data_uri_or_b64
        raw = base64.b64decode(b64_str, validate=False)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


def call_api(
    image_path: str,
    gender: str,
    reference_object: str,
    is_wall: bool,
    return_vis: bool,
) -> Tuple[Union[dict, str], Optional[Image.Image]]:
    """
    Returns: (json_payload_or_error_string, PIL.Image or None)
    """
    if not image_path:
        return {"error": "Please upload an image first."}, None

    filename = os.path.basename(image_path)
    mime = mimetypes.guess_type(filename)[0] or "image/jpeg"
    ref_obj = "paper_a4" if reference_object == "a4" else "paper_letter"

    try:
        with open(image_path, "rb") as f:
            files = {"file": (filename, f, mime)}
            data = {
                "gender": gender,
                "ref_obj": ref_obj,
                # booleans as 'true'/'false' strings so FastAPI parses them
                "is_wall": str(bool(is_wall)).lower(),
                "return_vis": str(bool(return_vis)).lower(),
            }
            r = requests.post(API_URL, files=files, data=data, timeout=45)

        # Raise for status to handle 4xx/5xx
        r.raise_for_status()

        # Try JSON first
        try:
            payload = r.json()
        except Exception:
            # Fallback if server returned non-JSON text
            return {"error": "Non-JSON response", "body": r.text}, None

        # Extract image if present
        vis_img = None
        img_b64 = payload.get("image_vis_b64")
        if img_b64:
            vis_img = _decode_b64_image(img_b64)

        return payload, vis_img

    # HTTP errors with optional JSON body
    except requests.HTTPError as e:
        try:
            body = e.response.json()
        except Exception:
            body = getattr(e.response, "text", "")
        return {"error": f"HTTP {e.response.status_code}", "body": body}, None

    # Any other client-side error
    except Exception as e:
        return {"error": f"Client error: {repr(e)}"}, None


with gr.Blocks() as demo:
    gr.Markdown("# Foot Size UI")

    def_ref_obj = "letter"
    if is_a4_format():
        def_ref_obj = "a4"

    with gr.Row():
        img = gr.Image(type="filepath", label="Foot photo")
        with gr.Column():
            gender = gr.Radio(["m", "f"], value="m", label="Gender")
            ref = gr.Radio(["letter", "a4"], value=def_ref_obj, label="Reference object")
            is_wall = gr.Checkbox(value=True, label="Foot touching wall")
            return_vis = gr.Checkbox(value=True, label="Return visualization")

    with gr.Row():
        out_json = gr.JSON(label="API response (JSON)")
        out_img = gr.Image(label="Visualization", interactive=False)

    btn = gr.Button("Measure", variant="primary")
    btn.click(
        fn=call_api,
        inputs=[img, gender, ref, is_wall, return_vis],
        outputs=[out_json, out_img],
    )

    with gr.Row():
        out_json = gr.JSON(label="API response (JSON)")
        out_img = gr.Image(label="Visualization", interactive=False)

    if AFFILIATE_URL:
        gr.HTML(f"""
                <div style="margin-top:0.5rem">
                <a href="{AFFILIATE_URL}"
                    target="_blank"
                    rel="nofollow sponsored noopener"
                    style="display:inline-block;padding:8px 12px;border-radius:10px;border:1px solid #ccc;text-decoration:none;">
                    🛒 Find running shoes on Amazon
                </a>
                <div style="font-size:12px;color:#666;margin-top:4px">
                    *As an Amazon Associate, I may earn from qualifying purchases.*
                </div>
                </div>
                """)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_api=False,
        inbrowser=False,
        share=False,
    )


# import os
# import io
# import mimetypes
# import requests

# import gradio as gr


# API_BASE = os.environ.get("API_BASE", "https://foot-size-api-762504128529.northamerica-northeast1.run.app")
# API_URL = f"{API_BASE}/measure"


# def call_api(image_path: str, gender: str, reference_object: str, is_wall, return_vis):
#     if not image_path:
#         return "Please upload an image first."

#     filename = os.path.basename(image_path)
#     mime = mimetypes.guess_type(filename)[0] or "image/jpeg"
#     ref_obj = "paper_a4" if reference_object == "a4" else "paper_letter"

#     try:
#         with open(image_path, "rb") as f:
#             files = {"file": (filename, f, mime)}
#             data = {"gender": gender,
#                     "ref_obj": ref_obj,
#                     "is_wall": str(bool(is_wall)).lower(),
#                     "return_vis": str(bool(return_vis)).lower()}
            
#             r = requests.post(API_URL, files=files, data=data, timeout=30)
#         r.raise_for_status()
#         try:
#             return r.json()
#         except Exception:
#             return r.text
#     except requests.HTTPError as e:
#         try:
#             return {"error": f"HTTP {e.response.status_code}", "body": e.response.json()}
#         except Exception:
#             return f"HTTP {e.response.status_code}: {getattr(e.response,'text','')}"
#     except Exception as e:
#         return f"Client error: {e!r}"


# with gr.Blocks() as demo:
#     gr.Markdown("# Foot Size UI")
#     img = gr.Image(type="filepath", label="Foot photo")  # <- returns str path
#     gender = gr.Radio(["m","f"], value="m", label="Gender")
#     ref = gr.Radio(["letter","a4"], value="letter", label="Reference object")
#     is_wall = gr.Checkbox(value=True, label="Foot touching wall")
#     return_vis = gr.Checkbox(value=False, label="Return visualization")
#     out = gr.Textbox(label="Response", lines=8)
#     btn = gr.Button("Measure")
#     btn.click(call_api, [img, gender, ref, is_wall, return_vis], out)


# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", "8080"))
#     demo.launch(server_name="0.0.0.0", server_port=port, show_api=False,
#                 inbrowser=False, share=False)
