import os
import shutil
import subprocess
import string

cover_in = "cover"
dataset = "dataset"

cover_out = os.path.join(dataset, "cover")
stego_out = os.path.join(dataset, "stego")

secrets = "secrets"


def generate_p(n):
    chars = string.ascii_lowercase

    if n <= len(chars):
        return chars[:n]

    repeats = (n // len(chars)) + 1
    return (chars * repeats)[:n]


def getSec():
    return [
        os.path.join(secrets, f)
        for f in os.listdir(secrets)
        if os.path.isfile(os.path.join(secrets, f))
    ]


def embed(image, secret, out_path, p):
    try:
        cmd = [
            "steghide",
            "embed",
            "-cf", image,
            "-ef", secret,
            "-sf", out_path,
            "-p", p
        ]

        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        return True

    except subprocess.CalledProcessError as e:
        print("I cant run it", e)
        return False


def build_ds():
    os.makedirs(cover_out, exist_ok=True)
    os.makedirs(stego_out, exist_ok=True)

    images = [
        f for f in os.listdir(cover_in)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    secret_files = getSec()
    print("images:", len(images))
    print("secrets:", len(secret_files))
    count = 1
    for sec in secret_files:
        secretName = os.path.splitext(os.path.basename(sec))[0]
        for img in images:
            imgName = os.path.splitext(img)[0]
            stego_name = f"stego_{secretName}_{imgName}.jpg"
            stego_path = os.path.join(stego_out, stego_name)
            src_img = os.path.join(cover_in, img)
            p = generate_p(count)
            shutil.copy2(src_img, os.path.join(cover_out, img))
            success = embed(src_img, sec, stego_path, p)
            if success:
                print(f"OK... who cares {img} + {secretName} | password={p}")
            else:
                print(f"Fail >> {img} + {secretName}")
            count += 1


if __name__ == "__main__":
    build_ds()