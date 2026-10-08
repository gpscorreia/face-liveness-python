import cv2
import mediapipe as mp
import time
import random
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

    current_action = ""
    action_frames = 0

    # Configuração do Jogo
    possible_actions = [
        {"id": "blink", "text": "PISQUE"},
        {"id": "turn_left", "text": "Vire a cabeça para a ESQUERDA"},
        {"id": "turn_right", "text": "Vire a cabeça para a DIREITA"}
    ]
    game_state = "waiting" # waiting, playing, game_over
    score = 0
    start_time = 0
    current_target = None
    last_processed_action = None

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

        action_detected = None

        if detection_result.face_landmarks:
            cv2.putText(frame, "Face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Etapa 3: Detecção de piscada (Blink)
            if detection_result.face_blendshapes:
                # Pegamos os blendshapes do primeiro rosto detectado
                blendshapes = detection_result.face_blendshapes[0]
                blink_left = 0.0
                blink_right = 0.0
                
                for category in blendshapes:
                    if category.category_name == 'eyeBlinkLeft':
                        blink_left = category.score
                    elif category.category_name == 'eyeBlinkRight':
                        blink_right = category.score
                
                # Se a pontuação de "olho fechado" for maior que 0.4 em ambos os olhos, detectamos a piscada
                if blink_left > 0.4 and blink_right > 0.4:
                    action_detected = "blink"
                    current_action = "Blink detected"
                    action_frames = 15

            # Etapa 4: Detecção de movimento da cabeça
            # Usaremos uma heurística geométrica simples: proporção da distância entre o nariz e as bochechas
            first_face_landmarks = detection_result.face_landmarks[0]
            nose_x = first_face_landmarks[1].x
            left_cheek_x = first_face_landmarks[234].x
            right_cheek_x = first_face_landmarks[454].x

            dist_left = nose_x - left_cheek_x
            dist_right = right_cheek_x - nose_x

            if dist_right > 0 and dist_left > 0:
                ratio = dist_left / dist_right
                # Se a distância esquerda é muito maior que a direita, o rosto virou para a esquerda (espelhado)
                if ratio > 1.8:
                    action_detected = "turn_right"
                    current_action = "Head turned RIGHT"
                    action_frames = 15
                # Se a distância direita é muito maior, o rosto virou para a esquerda
                elif ratio < 0.55:
                    action_detected = "turn_left"
                    current_action = "Head turned LEFT"
                    action_frames = 15

            # Desenhar os pontinhos no rosto
            for face_landmarks in detection_result.face_landmarks:
                for landmark in face_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Desenhar a última ação detectada por alguns frames (efeito memória)
        if action_frames > 0:
            color = (0, 255, 255) if "Blink" in current_action else (255, 0, 0)
            cv2.putText(frame, current_action, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            action_frames -= 1

        # Lógica do Jogo de Liveness
        if game_state == "waiting":
            cv2.putText(frame, "Pressione 's' para iniciar o Jogo!", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        elif game_state == "playing":
            time_left = 15 - int(time.time() - start_time)
            
            if time_left <= 0:
                game_state = "game_over"
            else:
                cv2.putText(frame, f"Tempo: {time_left}s", (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Pontos: {score}", (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Desafio: {current_target['text']}", (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

                # Avaliar ação detectada, ignorando se for a mesma ação da frame anterior 
                # (exige que volte ao rosto neutro antes de contar novamente)
                if action_detected and action_detected != last_processed_action:
                    if action_detected == current_target["id"]:
                        score += 1
                        current_target = random.choice(possible_actions)
                    else:
                        score = 0
                        current_target = random.choice(possible_actions)
                    last_processed_action = action_detected
                elif not action_detected:
                    last_processed_action = None
        elif game_state == "game_over":
            cv2.putText(frame, "TEMPO ESGOTADO!", (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(frame, f"Pontuacao Final: {score}", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, "Pressione 's' para jogar de novo.", (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Exibir a imagem
        cv2.imshow('Face Liveness MVP', frame)

        # Aguardar teclas
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and game_state != "playing":
            game_state = "playing"
            score = 0
            start_time = time.time()
            current_target = random.choice(possible_actions)
            last_processed_action = None

    # Liberar recursos
    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
