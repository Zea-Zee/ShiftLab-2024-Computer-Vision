import imageio
import os
from natsort import natsorted


output_dir = ""
image_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]


image_files = natsorted(image_files)

images = []
for image_file in image_files:
    image_path = os.path.join(output_dir, image_file)
    image = imageio.imread(image_path)
    images.append(image)

imageio.mimsave('', images, fps=25)
