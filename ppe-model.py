from roboflow import Roboflow
from ultralytics import YOLO
import os

rf = Roboflow(api_key="lo5aEdz34U40K023Ynha")
project = rf.workspace("muhammets-workspace-bufij").project("ppe-fall-detecetion-owg28")
version = project.version(1)
dataset = version.download("yolov8") 

print(f"Veri seti şu konuma indirildi: {dataset.location}")

model = YOLO('yolov8n.pt')

yaml_path = f"{dataset.location}/data.yaml"

print("Eğitim başlıyor...")
results = model.train(
    data=yaml_path,
    epochs=1,
    imgsz=320,
    batch=4,
    project='PPE_Model',
    name='test_egitimi',
    device='cpu',
    workers=0 # Reduce worker overhead for Windows
)

print("Eğitim tamamlandı! Sonuçlar 'PPE_Model/test_egitimi' klasörüne kaydedildi.")
