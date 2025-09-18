from time import sleep
from datetime import datetime
from sh import gphoto2 as gp
import signal, os, subprocess
from typing import Union
from fastapi import FastAPI
from fastapi.responses import FileResponse
import cv2
import numpy as np
import aiofiles



app = FastAPI()





def createSaveFolder(folder_name):
    try:
        os.makedirs(folder_name)
    except:
        print("Failed to create the new directory.")
    os.chdir(folder_name)
    print("Changed to directory: " + folder_name)

def captureImages(command, picID):
    gp(command)
    print("Captured the image: "+picID+".jpg")

async def grade_using_cv(filepath: str):
    """
    Mendeteksi dan mengklasifikasikan aflatoksin pada gambar jagung.
    - Warna isian sesuai dengan grade piksel masing-masing.
    - Warna kotak dan label sesuai dengan grade terparah dalam satu area.
    """
    print("Detecting and Grading Aflatoxin using OpenCV...")
    image = cv2.imread(str(filepath))
    if image is None:
        print(f"Error: Image not found at path: {filepath}")
        raise ValueError("Image not found or the path is incorrect")

    # 1. Pre-processing Gambar
    filtered_image = cv2.medianBlur(image, 5)
    filtered_image = cv2.GaussianBlur(filtered_image, (9, 9), 0)

    # 2. Kalkulasi Indeks NDFI
    B, G, R = cv2.split(filtered_image)
    B = B.astype(float)
    G = G.astype(float)
    NDFI = (B - G) / (B + G + 0.0001)
    NDFI_normalized = ((NDFI + 1.0) * 127.5).astype(np.uint8)
    
    # 3. Definisikan Tingkat Intensitas (PENTING: Urutkan dari terparah ke teringan)
    INTENSITY_LEVELS = {
        "REJECT (Sangat Terang)": {"range": (0, 150), "color": (0, 0, 255), "area": 0.0, "count": 0},
        "GRADE D (Terang)": {"range": (151, 160), "color": (0, 165, 255), "area": 0.0, "count": 0},
        "GRADE C (Redup)": {"range": (161, 168), "color": (0, 255, 255), "area": 0.0, "count": 0}
    }

    labeled_image = image.copy()
    min_contour_area = 80

    # --- LOGIKA BARU: DIPISAH MENJADI 2 TAHAP ---

    # TAHAP 1: KLASIFIKASI & PEWARNAAN PIKSEL (SEGMENTASI)
    # ----------------------------------------------------
    # Buat mask untuk setiap level secara eksklusif agar tidak tumpang tindih
    level_masks = {}
    processed_mask = np.zeros(NDFI_normalized.shape, dtype=np.uint8)
    for level_name, prop in INTENSITY_LEVELS.items():
        # Buat mask mentah untuk rentang warna saat ini
        raw_mask = cv2.inRange(NDFI_normalized, prop["range"][0], prop["range"][1])
        # Hapus piksel yang sudah diproses oleh level yang lebih tinggi
        exclusive_mask = cv2.bitwise_and(raw_mask, cv2.bitwise_not(processed_mask))
        level_masks[level_name] = exclusive_mask
        
        # Warnai isian (segmentasi) pada gambar output sesuai warna levelnya
        labeled_image[exclusive_mask > 0] = prop["color"]
        
        # Hitung statistik area dan jumlah untuk level ini
        prop["area"] = np.count_nonzero(exclusive_mask)
        contours, _ = cv2.findContours(exclusive_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        prop["count"] = len([c for c in contours if cv2.contourArea(c) > min_contour_area])

        # Perbarui master mask untuk iterasi selanjutnya
        processed_mask = cv2.bitwise_or(processed_mask, exclusive_mask)

    # TAHAP 2: DETEKSI & PELABELAN OBJEK
    # -------------------------------------
    # Gabungkan semua mask untuk menemukan objek/pulau kontaminasi yang utuh
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_detected_area2 = 0
    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            continue
        
        # Tentukan grade tertinggi di dalam kontur objek ini
        priority_level_name = None
        # Buat mask sementara hanya untuk kontur saat ini
        contour_mask_temp = np.zeros_like(processed_mask)
        cv2.drawContours(contour_mask_temp, [contour], -1, 255, cv2.FILLED)

        # Cek dari grade terparah ke teringan
        for level_name in INTENSITY_LEVELS.keys():
            # Cek apakah ada irisan antara kontur ini dengan mask level tersebut
            if np.any(cv2.bitwise_and(contour_mask_temp, level_masks[level_name])):
                priority_level_name = level_name
                break # Ditemukan level tertinggi, hentikan pencarian

        if priority_level_name:
            properties = INTENSITY_LEVELS[priority_level_name]
            priority_color = properties["color"]
            label = f'{priority_level_name.split(" ")[0]}'
            
            x, y, w, h = cv2.boundingRect(contour)
            # Gambar KOTAK dan TULISAN dengan warna prioritas tertinggi
            cv2.rectangle(labeled_image, (x, y), (x + w, y + h), priority_color, 2) 
            cv2.putText(labeled_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, priority_color, 2)
            total_detected_area2+=1

    # 5. Kalkulasi Total dan Tentukan Grade Final
    total_detected_area = sum(level["area"] for level in INTENSITY_LEVELS.values())
    height, width, _ = image.shape
    total_image_area = height * width
    percentage = (total_detected_area / total_image_area) * 100
    
    final_grade = "GRADE A (Bersih)"
    if INTENSITY_LEVELS["REJECT (Sangat Terang)"]["area"] > 0:
        final_grade = "REJECT"
    elif INTENSITY_LEVELS["GRADE D (Terang)"]["area"] > 0:
        final_grade = "GRADE D"
    elif INTENSITY_LEVELS["GRADE C (Redup)"]["area"] > 0:
        final_grade = "GRADE C"
    elif total_detected_area > 0:
        final_grade = "GRADE B (Kontaminasi Minor)"
    
    # 6. Tambahkan Informasi Grading ke Gambar
    info_text_y = 30
    cv2.putText(labeled_image, f"Final Grade: {final_grade}", (10, info_text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    info_text_y += 30
    cv2.putText(labeled_image, f"Total Area Terdeteksi: {total_detected_area:.2f} px ({percentage:.4f}%)", (10, info_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 7. Simpan Gambar Hasil
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_path = os.path.join("./hasil/", f"graded_image_cv-{timestamp}.jpg")
    
    _, img_encoded = cv2.imencode('.jpg', labeled_image)
    async with aiofiles.open(save_path, "wb") as img_file:
        await img_file.write(img_encoded.tobytes())
    print(f"✅ Graded image successfully saved to {save_path}")

    # 8. Siapkan data untuk respons API
    response_data = {
        "final_grade": final_grade,
        "total_area_pixels": total_detected_area,
        "total_area_percentage": percentage,
        "detection_details": {
            level: {
                "area": properties["area"],
                "count": properties["count"]
            } for level, properties in INTENSITY_LEVELS.items()
        },
        "total_detected_objects": total_detected_area2,
        "graded_image_path": save_path,
        "original_image_path": filepath
    }
    
    return response_data

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/captureImage")
def read_root():
    shot_date = datetime.now().strftime("%Y-%m-%d")
    shot_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    picID = "PiShots_" + shot_time

    captureAndDownloadCommand = ["--capture-image-and-download","--filename",picID+".jpg"]

    folder_name = shot_date
    createSaveFolder(folder_name)
    captureImages(captureAndDownloadCommand, picID)

    return FileResponse(picID+".jpg")


@app.get("/gradeImage/{image_path}")
async def grade_image(image_path: str):
    try:
        result = await grade_using_cv(image_path)
        return result
    except Exception as e:
        return {"error": str(e)}