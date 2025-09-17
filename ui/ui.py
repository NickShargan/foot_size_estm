import os
import io
import requests

# import gradio as gr
import os, mimetypes, requests, gradio as gr


API_URL = "https://foot-size-api-762504128529.northamerica-northeast1.run.app/measure"


def call_api(image_path: str, gender: str, reference_object: str, is_wall, return_vis):
    if not image_path:
        return "Please upload an image first."

    filename = os.path.basename(image_path)
    mime = mimetypes.guess_type(filename)[0] or "image/jpeg"
    ref_obj = "paper_a4" if reference_object == "a4" else "paper_letter"

    try:
        with open(image_path, "rb") as f:
            files = {"file": (filename, f, mime)}
            data = {"gender": gender,
                    "ref_obj": ref_obj,
                    "is_wall": str(bool(is_wall)).lower(),
                    "return_vis": str(bool(return_vis)).lower()}
            
            r = requests.post(API_URL, files=files, data=data, timeout=30)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return r.text
    except requests.HTTPError as e:
        try:
            return {"error": f"HTTP {e.response.status_code}", "body": e.response.json()}
        except Exception:
            return f"HTTP {e.response.status_code}: {getattr(e.response,'text','')}"
    except Exception as e:
        return f"Client error: {e!r}"


with gr.Blocks() as demo:
    gr.Markdown("# Foot Size UI")
    img = gr.Image(type="filepath", label="Foot photo")  # <- returns str path
    gender = gr.Radio(["m","f"], value="m", label="Gender")
    ref = gr.Radio(["letter","a4"], value="letter", label="Reference object")
    is_wall = gr.Checkbox(value=True, label="Foot touching wall")
    return_vis = gr.Checkbox(value=False, label="Return visualization")
    out = gr.Textbox(label="Response", lines=8)
    btn = gr.Button("Measure")
    btn.click(call_api, [img, gender, ref, is_wall, return_vis], out)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    demo.launch(server_name="0.0.0.0", server_port=port, show_api=False,
                inbrowser=False, share=False)
