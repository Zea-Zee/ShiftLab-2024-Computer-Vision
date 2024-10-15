import imageio
import os


output_dir = "MNIST-GAN-EVO_2024-10-15_18-54-29"
image_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])

images = []
for image_file in image_files:
    image_path = os.path.join(output_dir, image_file)
    image = imageio.imread(image_path)
    images.append(image)

imageio.mimsave('training_evolution.gif', images, fps=10)
