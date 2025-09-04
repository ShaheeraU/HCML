from face_recognition_system import FaceRecognitionSystem

# Initialize system
fr_system = FaceRecognitionSystem()


if fr_system.test_model():
    print("Model test passed successfully.")
    
    # Register users
    fr_system.register_user("Pritesh", "data/prit_sample1.jpeg")
    fr_system.register_user(
        "Chris", "data/chris_hemsworth_sample1.jpg")

    # Match a face
    user_id, name, similarity = fr_system.match_face(
        "data/chris_hemsworth_sample2.jpg", threshold=0.3)

    if user_id:
        print(f"Recognized: {name}")
    else:
        print("No match found")
else:
    print("Model test failed.")
