import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
# Import metric calculations
from tensorflow.keras.metrics import Precision, Recall

ANC_PATH = os.path.join("data", "anchor")
NEG_PATH = os.path.join("data", "negative")
POS_PATH = os.path.join("data", "positive")

anchor = tf.data.Dataset.list_files(ANC_PATH + '/*.jpg').take(500)
positive = tf.data.Dataset.list_files(POS_PATH + '/*.jpg').take(500)
negative = tf.data.Dataset.list_files(NEG_PATH + '/*.jpg').take(500)


def preprocess(file_path):
    # read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)

    # Preprocessing steps by resizing image 100,100,3
    img = tf.image.resize(img, (100, 100))
    # Scale image to be between 0 and 1
    img = img / 255.0

    return img


positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)


def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)


# Build dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

# Training partition
train_data = data.take(round(len(data) * .8))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# Testing partition
test_data = data.skip(round(len(data) * .8))

test_data = test_data.take(round(len(data) * .2))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


def make_embedding():
    inp = Input(shape=(100, 100, 3), name="input_image_layer")

    # First block
    c1 = Conv2D(64, (10, 10), activation="relu")(inp)
    m1 = MaxPooling2D(64, (2, 2), padding="same")(c1)

    # Second block
    c2 = Conv2D(128, (7, 7), activation="relu")(m1)
    m2 = MaxPooling2D(64, (2, 2), padding="same")(c2)

    # Third block
    c3 = Conv2D(128, (4, 4), activation="relu")(m2)
    m3 = MaxPooling2D(64, (2, 2), padding="same")(c3)

    # Final embedding block
    c4 = Conv2D(256, (4, 4), activation="relu")(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation="sigmoid")(f1)

    return Model(inputs=[inp], outputs=[d1], name="embedding_model")


# Siamese L1 distance class
class L1Dist(Layer):

    # init method - Inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


siamese_layer = L1Dist()


def make_siamese_model():
    # Anchor image input in the network
    input_image = Input(name="Input_img", shape=(100, 100, 3))

    # Validation image in the network
    validation_image = Input(name="validation_img", shape=(100, 100, 3))

    embedding = make_embedding()

    # Combine siamese distance layer components

    siamese_layer._name = "distance"
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    # Classification layer
    classifier = Dense(1, activation="sigmoid")(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name="siameseNetwork")

siamese_model = make_siamese_model()
# Reload model
model = tf.keras.models.load_model('siamesemodel.h5',
                                   custom_objects={'L1Dist': L1Dist,
                                                   'BinaryCrossentropy': tf.losses.BinaryCrossentropy})


def verify(frame, model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(os.path.join("application_data", "verification_images")):
        input_img = preprocess(os.path.join("application_data", "input_image", "input_image.jpg"))
        validation_img = preprocess(os.path.join("application_data", "verification_images", image))

        result = model.predict(list(np.expand_dims([input_img, validation_img],axis=1)))
        results.append(result)

    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(os.listdir(os.path.join("application_data", "verification_images")))
    verified = verification > verification_threshold

    return results, verified

    # Detection THreshold: Metric above which a prediction is considered positive
    # Verification THreshold: Propostion of positive predictions / total positive samples


cap = cv2.VideoCapture(1)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120:120 + 250, 200:200 + 250, :]

    cv2.imshow("verification", frame)

    # Verification trigger
    if cv2.waitKey(10) & 0xFF == ord("v"):
        # Save input image
        cv2.imwrite(os.path.join("application_data", "input_image", "input_image.jpg"), frame)
        # Verification function
        results, verified = verify(frame, model, 0.5, 0.6)
        print(verified)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
