import os
import shutil
import subprocess

cover_in = "cover"
dataset="dataset"

cover_out=os.path.join(dataset,"cover")
stego_out=os.path.join(dataset,"stego")

secrets="secrets"

def getSec():
    return [
        os.path.join(secrets,f) for f in os.listdir(secrets) if os.path.isfile(os.path.join(secrets))
    ]

def embed(image,secret,out_path):
    try :
        cmd=["steghide","embed","-cf",image,"-ef",secret,"-sf",out_path,"-p",""]
        subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        print("I cant run it ",e  )

def build_ds ():
    images=[f for f in os.listdir(cover_in) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    secrets=getSec()

    print("images:",len(images))
    print("secrets:",len(secrets))

    for i,sec in enumerate(secrets):
        secretName=os.path.splitext(os.path.basename(sec))[0]

        for j,img in enumerate(images):
            stego_name= f"stego_{secretName}.jpg"
            stego_path = os.path.join(stego_out,stego_name)
            src_img=os.path.join(cover_in,img)
            shutil.copy2(src_img,os.path.join(cover_out,img))
            success=embed(src_img,sec,stego_path)
            if success:
                print(f"OK.. who cares {img}+{secretName}")
            else:print(f"Fail >> {img} + {secretName}")


if __name__=="main":
    build_ds()



