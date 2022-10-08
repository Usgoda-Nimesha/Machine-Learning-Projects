import numpy as np
from keras.preprocessing import image
from keras.applications import resnet50

# load keras' ResNet50 model that was pre-trained against the ImageNet database
model = resnet50.ResNet50()

# load the image file, resizing it to 224*224 pixels(required by this model)
img = image.load_img("dog.jpg", target_size=(224, 224))

# convert the image to a numpy array
x = image.img_to_array(img)

# add a forth dimension since keras expects a list of images
x = np.expand_dims(x, axis=0)

# scale the input image to the range used in the trained network
x = resnet50.preprocess_input(x)

# Run the image through the deep neural network to make a prediction
predictions = model.predict(x)

# look up the names of predicted classes
predicted_clasess = resnet50.decode_predictions(predictions,top=5)

print("this is an image of: ")
for imagenet_id,name,likelihood in predicted_clasess[0]:
    print(" - {}: {:1f}".format(name,likelihood*100))

