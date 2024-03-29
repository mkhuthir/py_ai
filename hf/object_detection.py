#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

##----------------------------------------------------------
def render_output(raw_img, in_results):
    plt.figure(figsize=(16, 10))
    plt.imshow(raw_img)

    ax = plt.gca()

    for prediction in in_results:

        x, y    = prediction['box']['xmin'] , prediction['box']['ymin']
        w       = prediction['box']['xmax'] - prediction['box']['xmin']
        h       = prediction['box']['ymax'] - prediction['box']['ymin']

        ax.add_patch(plt.Rectangle((x, y),
                                   w,
                                   h,
                                   fill=False,
                                   color="green",
                                   linewidth=2))
        ax.text(x,
                y,
                f"{prediction['label']}: {round(prediction['score']*100, 1)}%",
                color='red'
        )

    plt.axis("off")

    # Save the modified image to a BytesIO object
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png',
                bbox_inches='tight',
                pad_inches=0)
    img_buf.seek(0)
    modified_image = Image.open(img_buf)

    # Close the plot to prevent it from being displayed
    plt.close()

    return modified_image
##----------------------------------------------------------

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # set tensorflow logs level

from transformers import pipeline 
from transformers.utils import logging
from PIL import Image
import matplotlib.pyplot as plt
import io

logging.set_verbosity_error()

pipe = pipeline("object-detection", 
                "facebook/detr-resnet-50",
                device=0)

raw_image = Image.open('../media/hf_friends.jpg')
raw_image.resize((569, 491))
output = pipe(raw_image)
output_image = render_output(raw_image,output)

raw_image.show()
output_image.show()
print(output)


