import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def get_face_landmarks(image):
    #Settings for face mesh
    with mp_face_mesh.FaceMesh(static_image_mode=False , max_num_faces=1 , min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(image)
        return results


cap = cv2.VideoCapture(0)

#The running loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # To flip image for user
    frame = cv2.flip(frame,1)
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = get_face_landmarks(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks: 

            # Drawing the face mesh
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)
            )


    cv2.imshow("Face_Mesh_program" , frame)
    if cv2.waitKey(5) & 0xFF == 27 or cv2.waitKey(5) & 0xFF == ord('q') :
        break

cap.release()
cv2.destroyAllWindows()
