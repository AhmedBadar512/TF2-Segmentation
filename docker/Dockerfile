# For now TF 2.13 has been tested
# TODO: Add support for the latest TF version
FROM nvcr.io/nvidia/tensorflow:23.10-tf2-py3
RUN pip install tensorflow_addons tensorflow_probability numpy opencv-python tqdm
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y