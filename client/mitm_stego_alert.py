"""
mitmproxy addon: watches BOTH uploaded and downloaded images
(request bodies and response bodies with Content-Type: image/*),
sends each to the stego-detection endpoint (/isnt/it/), and pops a
desktop alert if the result confirms or predicts a hidden payload.

Run with:
    mitmdump -s mitm_stego_alert.py

Desktop popups need one of:
    pip install plyer          (cross-platform, recommended)
or the OS-native fallback commands used below (notify-send / osascript / msg).
If none are available, it still prints a loud alert to the console.
"""

import time
import json
import requests
import subprocess
import platform
from mitmproxy import http

# Production endpoint: model + StegSeek confirmation for low-confidence
# JPEG "stego" calls. This is the ONLY endpoint the proxy should call.
STEGO_CHECK_URL = "http://10.25.125.170:8000/isnt/it/"
RESULTS_FILE = "stego_results.jsonl"
TIMEOUT = 10  # seconds


def send_to_stego_checker(filename: str, data: bytes) -> dict:
    try:
        resp = requests.post(
            STEGO_CHECK_URL,
            files={"image": (filename, data)},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def is_positive(response: dict) -> bool:
    """
    /isnt/it/ returns either:
      - a plain model result:      {"source": "model", "result": {...}}
      - a model + StegSeek result: {"source": "model+stegseek",
                                     "result": {...},
                                     "confirmed_stego": true/false}

    When StegSeek was involved, its confirmation is a deterministic,
    low-false-positive signal, so it takes priority over the model's own
    low-confidence guess. Otherwise fall back to the model's prediction.
    """
    if response.get("confirmed_stego") is True:
        return True
    if response.get("confirmed_stego") is False:
        # StegSeek explicitly failed to confirm; trust it over the
        # low-confidence model call that triggered the check.
        return False
    inner = response.get("result", {})
    return inner.get("prediction") == "stego"


def log_result(source_url: str, filename: str, result: dict):
    entry = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_url": source_url,
        "filename": filename,
        "result": result,
    }
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def show_alert(title: str, message: str):
    """
    Best-effort desktop popup, tries a few backends in order:
      1. plyer (cross-platform, if installed)
      2. OS-native command (Linux notify-send / macOS osascript / Windows msg)
      3. Loud console fallback (always works)
    """
    try:
        from plyer import notification
        notification.notify(title=title, message=message, timeout=8)
        return
    except Exception:
        pass

    system = platform.system()
    try:
        if system == "Linux":
            subprocess.run(["notify-send", title, message], check=False)
            return
        elif system == "Darwin":
            script = f'display alert "{title}" message "{message}"'
            subprocess.run(["osascript", "-e", script], check=False)
            return
        elif system == "Windows":
            subprocess.run(["msg", "*", f"{title}: {message}"], check=False)
            return
    except Exception:
        pass

    # Fallback: unmissable console alert
    banner = "!" * 60
    print(f"\n{banner}\n[STEGO ALERT] {title}\n{message}\n{banner}\n")


class InterceptStegoAddon:
    """Intercepts both directions of image traffic:
        request()  -> uploads   (client machine sending an image out)
        response() -> downloads (client machine receiving an image)
    """

    def request(self, flow: http.HTTPFlow):
        self._check_flow(flow, direction="upload")

    def response(self, flow: http.HTTPFlow):
        self._check_flow(flow, direction="download")

    def _check_flow(self, flow: http.HTTPFlow, direction: str):
        if direction == "upload":
            content_type = flow.request.headers.get("Content-Type", "")
            data = flow.request.content
            source_url = flow.request.pretty_url
        else:
            content_type = flow.response.headers.get("Content-Type", "")
            data = flow.response.content
            source_url = flow.request.pretty_url

        if not content_type.startswith("image/") or not data:
            return  # only raw image/* bodies are candidates (see Section 3.2
                     # of the technical report re: multipart/form-data uploads)

        ext = content_type.split("/")[-1].split(";")[0]
        if ext == "svg+xml":
            ext = "svg"
        filename = f"{int(time.time()*1000)}_{direction}.{ext}"

        result = send_to_stego_checker(filename, data)
        log_result(source_url, filename, result)

        if "error" in result:
            print(f"[stego-check] error checking {source_url}: {result['error']}")
            return

        print(f"[stego-check] {filename} ({direction}) <- {source_url} :: {result}")

        if is_positive(result):
            show_alert(
                "Steganography Detected!",
                f"{direction.capitalize()} image from {source_url} appears to contain a hidden payload.\n{result}",
            )


addons = [InterceptStegoAddon()]
