# PixelGuard AI

Network-level interception and deep-learning-based steganalysis of images in transit.

The system has two parts:

```
repo/
├── Model/
│   ├── server.py      # FastAPI detection service
│   └── SRnet.py        # SRNet deep learning model definition
└── client/
    └── Mitm_stego_alert.py   # mitmproxy addon (interception + alerting)
```

The **server** loads a trained SRNet checkpoint and exposes it over HTTP.
The **client** is a mitmproxy addon that intercepts image traffic (uploads
and downloads) on a machine, forwards each image to the server, and raises
a desktop alert if the server reports a hidden payload.

---

## 1. Requirements

**Server machine**

- Python 3.9+
- A trained checkpoint file named `srnet_last.pt` (place it in `Model/`, next to `server.py`)
- For JPEG confirmation: WSL (Windows Subsystem for Linux) with `stegseek`
  installed inside it — the server shells out to `wsl stegseek ...`. If your
  server isn't on Windows, see the note in Section 3.

```bash
cd Model
pip install fastapi uvicorn "python-multipart" pillow torch torchvision
```

**Client machine**

```bash
cd client
pip install mitmproxy requests
pip install plyer   # optional, for cross-platform desktop notifications
```

---

## 2. Running the server

From `Model/`:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

- `--host 0.0.0.0` makes it reachable from other machines on the network
  (needed since the client runs on a different machine). Use `127.0.0.1`
  instead if client and server are on the same machine.
- Check it's alive: open `http://<server-ip>:8000/` — you should get
  `{"status": 200}`.
- Uploaded images are also saved to `Model/uploads/` for reference.

### StegSeek / WSL note

`server.py` currently invokes StegSeek through WSL (`wsl stegseek --seed ...`).
This assumes the server runs on Windows with WSL installed and `stegseek`
available inside the WSL distro (`sudo apt install stegseek` on
Debian/Ubuntu-based WSL). If your server runs directly on Linux, replace the
`"wsl", "wslpath", ...` / `"wsl", "stegseek", ...` subprocess calls in
`server.py` with a direct call to `stegseek` — the WSL hop is unnecessary
outside a Windows host.

---

## 3. Running the client

From `client/`, first open `Mitm_stego_alert.py` and set `STEGO_CHECK_URL`
to point at your server:

```python
STEGO_CHECK_URL = "http://<server-ip>:8000/isnt/it/"
```

Then start mitmproxy with the addon loaded:

```bash
mitmdump -s Mitm_stego_alert.py
```

By default mitmproxy listens on port **8080**. You'll see console output
like `[stego-check] ... :: {...}` for every image it inspects, and results
are appended to `stego_results.jsonl` in the same folder.

---

## 4. Pointing Firefox at the proxy

Firefox needs two things configured: the proxy address, and trust for
mitmproxy's certificate (since mitmproxy has to decrypt HTTPS traffic to
inspect it, and Firefox keeps its own certificate store separate from the
OS).

### 4.1 Set the proxy

1. With `mitmdump` running, open Firefox → **Settings** → scroll to
   **Network Settings** → **Settings...**
2. Choose **Manual proxy configuration**.
3. Set **HTTP Proxy** to `127.0.0.1` (or the client machine's address if
   Firefox is on a different machine than mitmproxy) with **Port** `8080`.
4. Check **"Also use this proxy for HTTPS"** (older Firefox) — on newer
   versions a single HTTP proxy field covers both once you save.
5. Click **OK**.

### 4.2 Install the mitmproxy CA certificate

1. With the proxy active, visit **`http://mitm.it`** in Firefox.
2. Click the **Firefox** icon on that page and download the certificate.
3. Firefox will prompt to import it — alternatively, go to
   **Settings → Privacy & Security → Certificates → View Certificates →
   Authorities → Import**, and select the downloaded `mitmproxy-ca-cert.pem`.
4. When importing, check **"Trust this CA to identify websites"**.

Without this step, Firefox will show certificate warnings on every HTTPS
site and mitmproxy won't be able to see inside encrypted traffic.

### 4.3 Verify interception is working

Browse to any page with images. In the `mitmdump` console you should see
`[stego-check]` log lines firing for each image. If you see nothing:

- Confirm the proxy settings actually saved (revisit the Network Settings page).
- Confirm `mitmdump` is still running and listening on the port you configured.
- Confirm `STEGO_CHECK_URL` in `Mitm_stego_alert.py` matches the server's real
  address and port, and that the client machine can reach the server
  (`curl http://<server-ip>:8000/` from the client machine as a quick check).

To stop intercepting, switch Firefox's proxy setting back to **No proxy** or
**Use system proxy settings**.

---

## 5. What counts as "in scope"

The addon hooks both directions of traffic:

- `response()` → images being **downloaded** to the client machine
- `request()` → images being **uploaded** from the client machine

Only bodies with a `Content-Type: image/*` header are checked. Images
embedded inside a `multipart/form-data` upload (common with web upload
forms) are **not currently parsed out** and will be skipped — see the
technical report for details on this known limitation.

---

## 6. Desktop alerts

When the server confirms or predicts a hidden payload, the client tries,
in order: `plyer` notifications → OS-native commands
(`notify-send` on Linux, `osascript` on macOS, `msg` on Windows) → a
console banner as a last resort. At least one of these will always work,
so alerts don't silently disappear even on a machine with nothing extra
installed.
