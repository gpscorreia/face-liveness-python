import cv2
import mediapipe as mp

def main():
    # Inicializar MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Etapa 1: Abrir a webcam e exibir a imagem
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao acessar a webcam.")
        return

    print("Pressione 'q' para sair.")

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Não foi possível capturar o frame.")
                break

            # Espelho da imagem (inverte horizontalmente) para ficar mais natural
            frame = cv2.flip(frame, 1)

            # Converter a imagem BGR para RGB para o MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Etapa 2: Processar a detecção facial
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                cv2.putText(frame, "Face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Desenhar os landmarks principais para termos feedback visual
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )

            # Exibir a imagem
            cv2.imshow('Face Liveness MVP', frame)

            # Aguardar tecla 'q' para sair
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
