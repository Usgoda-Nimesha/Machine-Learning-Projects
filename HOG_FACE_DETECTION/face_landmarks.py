import PIL.Image
import PIL.ImageDraw
import face_recognition

# Load the jpg file into numpy array
image = face_recognition.load_image_file("people.jpg")

# Find all facial fetures
face_landmarks_list = face_recognition.face_landmarks(image)

print(len(face_landmarks_list))

# Load the image into a python Image Library object so that we can draw on top
pil_image = PIL.Image.fromarray(image)

# Create a PIL drawing object to be able to draw lines later
draw = PIL.ImageDraw.Draw(pil_image)

# Loop over each face
for face_landmarks in face_landmarks_list:
    # Loop over each facial feature
    for name, list_of_points in face_landmarks.items():
        print(name, list_of_points)
        draw.line(list_of_points, fill="red", width=2)

pil_image.show()
