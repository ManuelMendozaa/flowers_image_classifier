# Imports
import argparse
import numpy as np
import json
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

# Arguments
parse = argparse.ArgumentParser()
parse.add_argument('path_to_image')
parse.add_argument('saved_model')
parse.add_argument('--top_k', type=int, default=5)
parse.add_argument('--category_names', default='label_map.json')
args = parse.parse_args()

# Extract info
image = args.path_to_image
saved_model = args.saved_model
top_k = args.top_k
category_names = args.category_names

# Aux functions
def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    image = image.numpy()
    return image

def predict(image_path, model, top_k):
    # Open image
    X = Image.open(image_path)
    # Convert to numpy
    X = np.asarray(X)
    # Use process image function
    X = process_image(X)
    # Turn it into a sigle image batch
    X = np.expand_dims(X, axis = 0)
    # Predict
    prediction = model.predict(X)
    # Extract top k
    probs, classes = tf.math.top_k(prediction, k=top_k)
    return probs, classes

def print_results(top_k, classes):
    for i in range(top_k):
        class_name = category_names[str(classes.numpy()[0, i]+1)]
        prob = probs[0][i] * 100
    
        print(f'{class_name}: \t {prob}%')

# Main function
def main():
    # Get model
    model = tf.keras.models.load_model(saved_model)
    
    # Predict
    probs, classes = predict(image, model, top_k)
    
    # Show results
    print_results(probs, classes)
    

if __name__ == '__main__': 
    main()
    

