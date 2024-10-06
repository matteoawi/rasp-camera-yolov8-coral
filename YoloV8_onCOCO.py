import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
import time
from collections import defaultdict
from threading import Thread

# Thread per la cattura video per migliorare le prestazioni
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            if self.cap.isOpened():
                self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# Inizializza il modello YOLO con il task di rilevamento (detection)
model = YOLO('240_yolov8n_full_integer_quant_edgetpu.tflite', task='detect')

# Avvia il video stream in un thread separato
vs = VideoStream().start()

# Aggiungi un ritardo per dare il tempo alla fotocamera di inizializzarsi
time.sleep(3)

# Carica la lista delle classi da COCO
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

frame_count = 0
start_time = time.time()

# Variabile per tenere traccia del tempo totale impiegato per l'inferenza
inference_start_time = time.time()
total_inference_time = 0
inference_frame_count = 0

while True:
    ret, frame = vs.read()
    if not ret:
        print("Errore: impossibile catturare il frame")
        break

    frame_count += 1
    if frame_count % 6 != 0:
        continue  # Elabora ogni 3 frame per migliorare le performance

    # Riduci la dimensione del frame a 240p (320x240)
    resized_frame = cv2.resize(frame, (320, 240))

    # Inizio del timer per l'inferenza
    inference_start = time.time()

    # Esegui la predizione YOLO sul frame ridotto a 240p
    results = model.predict(resized_frame, imgsz=240)

    # Fine del timer per l'inferenza
    inference_end = time.time()
    total_inference_time += (inference_end - inference_start)
    inference_frame_count += 1

    # Dizionario per contare gli oggetti rilevati per ogni etichetta
    label_count = defaultdict(int)

    # Verifica che ci siano box di rilevamento
    if len(results) > 0:
        a = results[0].boxes.data
        if a is not None and len(a) > 0:
            px = pd.DataFrame(a).astype("float")

            # Loop sui risultati e disegna rettangoli intorno agli oggetti
            for index, row in px.iterrows():
                x1 = int(row[0] * (frame.shape[1] / 320))  # Converti le coordinate ridimensionate
                y1 = int(row[1] * (frame.shape[0] / 240))
                x2 = int(row[2] * (frame.shape[1] / 320))
                y2 = int(row[3] * (frame.shape[0] / 240))
                d = int(row[5])
                c = class_list[d]

                # Disegna rettangoli intorno agli oggetti rilevati
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

                # Aggiungi l'etichetta al contatore
                label_count[c] += 1

    # Calcola FPS
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time

    # Mostra gli FPS sul frame
    cvzone.putTextRect(frame, f'FPS (visualizzazione): {round(fps, 2)}', (10, 30), 1, 1)

    # Calcola e mostra il frame rate effettivo dell'inferenza
    if inference_frame_count > 0:
        inference_fps = inference_frame_count / total_inference_time
        cvzone.putTextRect(frame, f'FPS (inferenza): {round(inference_fps, 2)}', (10, 60), 1, 1)

    # Mostra il conteggio delle etichette rilevate
    y_offset = 90  # Posizione iniziale per il testo
    for label, count in label_count.items():
        cvzone.putTextRect(frame, f'{count} {label}', (10, y_offset), 1, 1)
        y_offset += 30  # Sposta il testo verso il basso per ogni nuova etichetta

    # Visualizza il frame
    cv2.imshow("FRAME", frame)

    # Interrompi l'esecuzione con ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

vs.stop()
cv2.destroyAllWindows()
