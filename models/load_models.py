import os
import tensorflow as tf

def download_model(drive_id, output_path):
    import gdown
    url = f"https://drive.google.com/uc?id={drive_id}"
    gdown.download(url, output_path, quiet=False)

def load_tomato_model():
    model_path = "models/tomato_ready_model10.h5"
    if not os.path.exists(model_path):
        download_model("1DhXwbRhjWV2qpBDD0D3AmVSIufqlgP9R", model_path)
    return tf.keras.models.load_model(model_path)

def load_banana_model():
    model_path = "models/banana_ripeness_model20.h5"
    if not os.path.exists(model_path):
        download_model("1DYOcwfXVVSMy1yYQWAnSGfx_2uAwZxGA", model_path)
    return tf.keras.models.load_model(model_path)

tomato_model = load_tomato_model()
banana_model = load_banana_model()

tomato_img_size = (224, 224)
banana_img_size = (150, 150)
