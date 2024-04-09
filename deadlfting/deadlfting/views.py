

# ======================================working================================================================

import tensorflow as tf  #import Keras
import cv2
import base64  # receive the image data
import json
from channels.generic.websocket import WebsocketConsumer

# ============================model predicting with image saving==================================================================
import os
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from keras.models import load_model

# Load the model

model = tf.keras.models.load_model('model/Deadlift_Model.h5')

@csrf_exempt
def image_save(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']

        # Read the uploaded image
        img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)

        # Resize to match the input shape expected by the model
        img = cv2.resize(img, (256, 256))
        img = img / 255.0  # Normalize pixel values

        # Expand dimensions to create a batch of size 1 (required by the model)
        img = np.expand_dims(img, axis=0)

        # Make predictions
        predictions = model.predict(img)

        # Interpret predictions
        class_names = ['Correct Pose','Incorrect Pose']
        predicted_class = class_names[np.argmax(predictions)]
        print("Predicted Class:", predicted_class)

        # Specify the directory to save the image
        image_dir = os.path.join('images', image.name)
        # Save the image to the specified directory
        with open(image_dir, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)
        return JsonResponse({'message': 'Image saved successfully', 'predicted_class': predicted_class}, status=200)
    else:
        return JsonResponse({'error': 'No image file received'}, status=400)

# ============================model predicting with image saving==================================================================


@csrf_exempt
def predict_pose(request):
    print(request.body)
    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body_data = json.loads(body_unicode)
        frame_data = body_data.get('frame', None)
        # print(frame_data)
        if frame_data:
            # Convert base64 image to numpy array
            nparr = np.frombuffer(base64.b64decode(frame_data.split(',')[1]), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Save the image as JPG (optional)
            # cv2.imwrite('input_image.jpg', img)

            # Resize and preprocess the image
            img = cv2.resize(img, (256, 256))
            img = img / 255.0

            # Expand dimensions to create a batch of size 1 (required by the model)
            img = np.expand_dims(img, axis=0)

            # Make predictions
            predictions = model.predict(img)

            # Interpret predictions
            class_names = ['Correct Pose','Incorrect Pose', ]
            predicted_class = class_names[np.argmax(predictions)]

            # Return the prediction result
            # print(predictions)
            print(predicted_class)
            return JsonResponse({'prediction': predicted_class})
        else:
            return JsonResponse({'error': 'No frame data received'}, status=400)
    else:
        return JsonResponse({'error': 'POST request expected'}, status=400)

