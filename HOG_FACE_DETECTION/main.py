import PIL.Image
import PIL.ImageDraw
import face_recognition

# Load the image into a numpy array
image = face_recognition.load_image_file("people.jpg")

# Find all the faces in the image
face_locations = face_recognition.face_locations(image)

number_of_faces = len(face_locations)
print(number_of_faces)

# Load the image into a python image library object so that we can draw on top of the image
pil_image = PIL.Image.fromarray(image)

for face_location in face_locations:
    top, right, bottom, left = face_location
    draw = PIL.ImageDraw.Draw(pil_image)
    draw.rectangle([left,top,right,bottom], outline="red")

pil_image.show()