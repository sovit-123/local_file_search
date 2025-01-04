"""
Utilities for the Gradio application.
"""

from PIL import Image

import cv2

def load_and_preprocess_images(image_path):
    """
    Loads the uploaded image for image chat.

    :param image_path: Image file path.

    Returns:
        image: PIL image.
    """
    image = Image.open(image_path)
    return image

def load_and_process_videos(file_path, images, placeholder, counter):
    """
    Loads and processes videos for the Phi-3 model chat.

    :param file_path: File path of the video.
    :param images: A list to store the frames.
    :param placeholder: A placeholder string to store frame counter token.
    :param counter: A counter keeping count of the number of frames.

    Returns:
        image: The list containing all frames.
        placeholder: The updated placeholder with the correct counter token
            information.
        counter: A counter integer containing the total frames.
    """
    cap = cv2.VideoCapture(file_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(length):
        counter += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if frame is not None:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(image))
            placeholder += f"<|image_{counter}|>\n"
    return images, placeholder, counter