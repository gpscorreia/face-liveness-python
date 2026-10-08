import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

def download_model():
    model_path = 'face_landmarker.task'
    if not os.path.exists(model_path):
        print("Baixando o modelo Face Landmarker do MediaPipe...")
        url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
        urllib.request.urlretrieve(url, model_path)
        print("Download concluído!")
    return model_path

def main():
    model_path = download_model()

    # Configuração do Face Landmarker da nova API (Tasks)
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1)
    
    detector = vision.FaceLandmarker.create_from_options(options)

    # Etapa 1: Abrir a webcam e exibir a imagem
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao acessar a webcam.")
        return

    print("Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Não foi possível capturar o frame.")
            break

        # Espelho da imagem (inverte horizontalmente) para ficar mais natural
        frame = cv2.flip(frame, 1)

        # Converter a imagem BGR para RGB para o MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # O MediaPipe Tasks requer um objeto de Imagem próprio
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Etapa 2: Processar a detecção facial
        detection_result = detector.detect(mp_image)

        if detection_result.face_landmarks:
            cv2.putText(frame, "Face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Desenhar os pontinhos no rosto
            for face_landmarks in detection_result.face_landmarks:
                for landmark in face_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Exibir a imagem
        cv2.imshow('Face Liveness MVP', frame)

        # Aguardar tecla 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
