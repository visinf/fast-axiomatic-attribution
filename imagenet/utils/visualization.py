# obtained from https://github.com/ankurtaly/Integrated-Gradients/blob/master/VisualizationLibrary/visualization_lib.py

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from io import BytesIO
from IPython.display import display
from IPython.display import Image as Img

def pil_img(a):
    a = np.uint8(a)
    return Image.fromarray(a)

def show_img(img, out_dir, name):
    img.save(out_dir + name)
    show_pil_image(img)

def visualize_attrs(img, attrs, ptile=99, out_dir='./', name='out.png'):
    pixel_attrs = np.sum(np.abs(attrs), axis=2)
    pixel_attrs = np.clip(pixel_attrs/np.percentile(pixel_attrs, ptile), 0,1)
    vis = img*np.transpose([pixel_attrs, pixel_attrs, pixel_attrs], axes=[1,2,0])
    show_img(pil_img(vis), out_dir, name)

def show_pil_image(pil_img):
  """Display the provided PIL image.
  Args:
    pil_img: (PIL.Image) The provided PIL image.
  """
  f = BytesIO()
  pil_img.save(f, 'png')
  display(Img(data=f.getvalue()))