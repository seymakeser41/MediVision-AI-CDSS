from ultralytics import YOLO

if __name__ == '__main__':
    # Proje yapına göre kaydedilen last.pt dosyasının yolu
    # (Genellikle project/name altındaki weights klasöründe bulunur)
    model = YOLO(r"C:\Users\Seyma\Desktop\Bitirme_veri_hazirlk\runs\segment\Medikal_Vision\yolo_lezyon_seg-2\weights\last.pt")

    # resume=True parametresi ile kaldığı yerden devam ettirilir
    results = model.train(resume=True)
    
    print("YOLO Eğitimi kaldığı yerden başarıyla sürdürülüyor...")