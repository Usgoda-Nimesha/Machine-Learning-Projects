import face_recognition

imgp1 = face_recognition.load_image_file("person_1.jpg")
imgp2 = face_recognition.load_image_file("person_2.jpg")
imgp3 = face_recognition.load_image_file("person_3.jpg")

p1_face_encoding = face_recognition.face_encodings(imgp1)[0]
p2_face_encoding = face_recognition.face_encodings(imgp2)[0]
p3_face_encoding = face_recognition.face_encodings(imgp3)[0]

known_face_encodings = [
    p1_face_encoding,
    p2_face_encoding,
    p3_face_encoding
]

# Load image we want to check
unknown_image = face_recognition.load_image_file("unknown_7.jpg")

# Get face encodings for any people in the picture
face_locations = face_recognition.face_locations(unknown_image,number_of_times_to_upsample=2)
unknown_face_encodings = face_recognition.face_encodings(unknown_image,known_face_locations=face_locations)

# THere might be more than one person in the image
for unknown_face_encoding in unknown_face_encodings:
    # Test if this unknown face encoding matches any of the three people we know
    results = face_recognition.compare_faces(known_face_encodings,unknown_face_encoding)
    name = "unknown"

    if results[0]:
        print("person 1")
    if results[1]:
        print("person 2")
    if results[2]:
        print("person 3")