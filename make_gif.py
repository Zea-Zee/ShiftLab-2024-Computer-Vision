import imageio
import os
from natsort import natsorted  # Импортируем natsorted

output_dir = "SIN-GAN-EVO_2024-10-15_19-54-00"
image_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]

# Используем natsorted для сортировки файлов
image_files = natsorted(image_files)

images = []
for i, image_file in enumerate(image_files):
    image_path = os.path.join(output_dir, image_file)
    image = imageio.imread(image_path)
    images.append(image)

imageio.mimsave('new_sinus_training_evolution.gif', images, fps=25)
