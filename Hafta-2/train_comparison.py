import os
import time
import pandas as pd
import torch
from ultralytics import YOLO
from roboflow import Roboflow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "lo5aEdz34U40K023Ynha") # Fallback to user's provided key
WORKSPACE = "muhammets-workspace-bufij"
PROJECT_ID = "ppe-fall-detecetion-owg28"
VERSION_NUMBER = 1

# Models to compare
# Format: (model_name, model_alias for reporting)
MODELS_TO_COMPARE = [
    # YOLOv8
    ("yolov8n.pt", "YOLOv8-Nano"),
    ("yolov8s.pt", "YOLOv8-Small"),
    ("yolov8m.pt", "YOLOv8-Medium"),
    # YOLOv11
    ("yolo11n.pt", "YOLO11-Nano"),
    ("yolo11s.pt", "YOLO11-Small"),
    ("yolo11m.pt", "YOLO11-Medium"),
    # YOLOv12
    ("yolo12n.pt", "YOLO12-Nano"),
    ("yolo12s.pt", "YOLO12-Small"),
    ("yolo12m.pt", "YOLO12-Medium"),
    # YOLO26
    ("yolo26n.pt", "YOLO26-Nano"),
    ("yolo26s.pt", "YOLO26-Small"),
    ("yolo26m.pt", "YOLO26-Medium"),
]

EPOCHS = 3 # Basit bir karşılaştırma için düşük tutuldu
IMGSZ = 640
BATCH = 8 # GPU için batch boyutu artırıldı

# GPU tespiti ve güvenli cihaz seçimi
try:
    if torch.cuda.is_available():
        DEVICE = 0
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        print(f"CUDA tespit edildi: {device_name} (sm_{capability[0]}{capability[1]})")
        
        # Test allocation to be 100% sure
        try:
            torch.cuda.init()
            # Simple tensor test to catch kernel image errors
            torch.zeros(1).cuda()
        except RuntimeError as e:
            if "no kernel image" in str(e).lower():
                print(f"\nUYARI: {device_name} (sm_{capability[0]}{capability[1]}) mevcut PyTorch kurulumu ile uyumlu değil!")
                print("Lütfen PyTorch'un Blackwell (sm_120) destekleyen sürümünü (Nightly) kullandığınızdan emin olun.")
            else:
                print(f"CUDA başlatılamadı: {e}")
            print("CPU'ya dönülüyor...")
            DEVICE = 'cpu'
    else:
        DEVICE = 'cpu'
        print("CUDA bulunamadı veya Torch CPU versiyonu yüklü. CPU kullanılacak.")
except Exception as e:
    print(f"GPU kontrolü sırasında hata: {e}. CPU'ya dönülüyor.")
    DEVICE = 'cpu'

def setup_dataset():
    # Mevcut veri setini kontrol et
    local_path = os.path.join("PPE", "Fall-Detecetion-1")
    local_yaml = os.path.join(local_path, "data.yaml")
    
    if os.path.exists(local_yaml):
        print(f"Veri seti zaten mevcut: {local_yaml}")
        return os.path.abspath(local_yaml)
    
    print("Veri seti bulunamadı, Roboflow'dan indiriliyor...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT_ID)
    version = project.version(VERSION_NUMBER)
    dataset = version.download("yolov8")
    return f"{dataset.location}/data.yaml"

def run_comparison(yaml_path):
    results_data = []
    
    # Mevcut sonuçları yükle (eğer varsa)
    csv_path = "comparison_results.csv"
    if os.path.exists(csv_path):
        try:
            old_df = pd.read_csv(csv_path)
            results_data = old_df.to_dict('records')
            print(f"Mevcut sonuçlar yüklendi: {len(results_data)} model bulundu.")
        except Exception as e:
            print(f"Eski sonuçlar okunurken hata oluştu: {e}")

    # Zaten eğitilmiş modellerin listesi
    completed_models = [r["Model"] for r in results_data]

    for model_file, model_alias in MODELS_TO_COMPARE:
        if model_alias in completed_models:
            print(f"ATLANDI: {model_alias} zaten eğitilmiş.")
            continue

        print(f"\n{'='*50}")
        print(f"Eğitim Başlatılıyor: {model_alias} ({model_file})")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        try:
            # Model dosyasını 'models' klasörü altında ara/indir
            model_path = os.path.join("models", model_file)
            model = YOLO(model_path)
            train_results = model.train(
                data=yaml_path,
                epochs=EPOCHS,
                imgsz=IMGSZ,
                batch=BATCH,
                project='Comparison_Study',
                name=model_alias,
                device=DEVICE,
                workers=0,
                verbose=False
            )
            
            end_time = time.time()
            total_duration = end_time - start_time
            
            # Extract metrics from validation (last epoch)
            metrics = train_results.results_dict
            
            summary = {
                "Model": model_alias,
                "mAP50": metrics.get("metrics/mAP50(B)", 0),
                "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0),
                "Training Time (s)": round(total_duration, 2),
                "Model Size": model_file.split('.')[-2][-1], # n, s, m
                "Version": "v" + model_file.replace("yolo", "").replace(".pt", "")[:1] if "yolo" in model_file else "v8"
            }
            
            results_data.append(summary)
            print(f"TAMAMLANDI: {model_alias} | mAP50: {summary['mAP50']:.4f} | Süre: {summary['Training Time (s)']}s")
            
        except Exception as e:
            print(f"HATA: {model_alias} ({model_file}) eğitilirken bir sorun oluştu!")
            print(f"Hata Detayı: {str(e)}")
            import traceback
            traceback.print_exc()

    # Save results to CSV if not empty
    if not results_data:
        print("\nUYARI: Hiçbir model başarıyla eğitilemedi. Tablo oluşturulamıyor.")
        return pd.DataFrame()
        
    df = pd.DataFrame(results_data)
    df.to_csv("comparison_results.csv", index=False)
    print("\nKarşılaştırma bitti! Sonuçlar 'comparison_results.csv' dosyasına kaydedildi.")
    return df

if __name__ == "__main__":
    print("Veri seti hazırlanıyor...")
    data_yaml = setup_dataset()
    print(f"Veri seti hazır: {data_yaml}")
    
    comparison_df = run_comparison(data_yaml)
    print("\n--- Özet Tablo ---")
    print(comparison_df.to_string(index=False))
