# Dosya Yolu: C:\DragonAI\server\main.py

import torch
import intel_extension_for_pytorch as ipex # Intel GPU Sihri
from diffusers import StableDiffusionPipeline
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

# CORS Ayarları: HTML dosyanın API'ye bağlanabilmesi için şart
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Çıktı klasörünü dış dünyaya (arayüze) açıyoruz
output_path = "C:/DragonAI/outputs"
if not os.path.exists(output_path):
    os.makedirs(output_path)
app.mount("/outputs", StaticFiles(directory=output_path), name="outputs")

# MODEL YÜKLEME (Intel Arc B580 Optimizasyonlu)
print("Model XPU (GPU) üzerine yükleniyor, lütfen bekleyin...")
model_id = "runwayml/stable-diffusion-v1-5" # İlk açılışta internetten indirir
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("xpu") # Intel Arc hızlandırması burada devreye giriyor

@app.post("/generate/{username}")
async def generate(username: str, prompt: str):
    try:
        # Kullanıcıya özel klasör kontrolü
        user_dir = os.path.join(output_path, username)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)

        # Görsel Üretimi
        with torch.xpu.amp.autocast(): # XMX Hızlandırması için otomatik hassasiyet
            image = pipe(prompt, num_inference_steps=25).images[0]

        # Dosya adını belirle (Tarih ve Random ID eklenebilir)
        filename = f"dragon_{os.urandom(3).hex()}.png"
        save_path = os.path.join(user_dir, filename)
        image.save(save_path)

        return {"status": "success", "image_url": f"http://127.0.0.1:8000/outputs/{username}/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{username}")
async def get_history(username: str):
    user_dir = os.path.join(output_path, username)
    if not os.path.exists(user_dir):
        return []
    return [f for f in os.listdir(user_dir) if f.endswith(('.png', '.jpg'))]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)