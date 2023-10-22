import tensorflow as tf
from keras.models import load_model
import tensorflow_hub as hub

# Load the saved model
new_model = tf.keras.models.load_model('handwritemodel.h5')

# Display the model summary
new_model.summary()
