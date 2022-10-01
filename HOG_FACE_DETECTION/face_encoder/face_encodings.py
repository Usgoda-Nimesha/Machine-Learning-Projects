import face_recognition

image = face_recognition.load_image_file("person.jpg")

# Generate face encodings
face_encodings = face_recognition.face_encodings(image)

if len(face_encodings) == 0:
    print("no faces found")

else:
    first_face_encoding = face_encodings[0]
    print(len(face_encodings))