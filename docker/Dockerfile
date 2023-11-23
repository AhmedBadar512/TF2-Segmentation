FROM tensorflow/tensorflow:2.13.0-gpu
RUN pip install tensorflow_addons tensorflow_probability numpy opencv-python tqdm
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y