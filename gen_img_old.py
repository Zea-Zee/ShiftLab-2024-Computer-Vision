def generate_text_image(text, img=None, noise=False, max_char_y_offset=0.025, max_char_x_offset=0.025):
    gen_height, gen_width = image_size, image_size * 4

    if img is None:
        background_color = np.random.randint(0, 256, size=3, dtype=np.uint8)
        img = np.ones((gen_height, gen_width, 3), dtype=np.uint8) * background_color

    # Выбираем случайный шрифт
    font_path = random.choice(config.custom_fonts_paths)
    font_size = random.randint(20, 35)
    font = ImageFont.truetype(font_path, font_size)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Вычисляем размер текста и его позицию
    bbox = draw.textbbox((0, 0), text, font=font)
    text_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    while text_size[0] > img.shape[1] or text_size[1] > img.shape[0]:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])

    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (img.shape[0] - text_size[1]) // 2

    # Цвет текста
    background_color = np.mean(img, axis=(0, 1)).astype(int).tolist()
    text_color = tuple([255 - c for c in background_color])

    for char in text:
        # Получаем размер текущей буквы
        bbox = draw.textbbox((text_x, text_y), char, font=font)
        char_width = bbox[2] - bbox[0]
        char_height = bbox[3] - bbox[1]

        x_offset = random.randint(round(char_width * (-max_char_x_offset)), round(char_width * max_char_x_offset) // 1)
        y_offset = random.randint(round(char_height * (-max_char_y_offset)) // 1, round(char_height * max_char_y_offset) // 1)
        draw.text((text_x + x_offset, text_y + y_offset), char, font=font, fill=text_color)

        text_x += char_width    # Сдвигаем на ширину текущей буквы

    # Преобразуем обратно в формат OpenCV
    img = np.array(pil_img)


    font_name = font_path.replace('\\', '/')
    font_name = font_name.replace('fonts/', '')
    font_name = font_name.replace('.ttf', '')


    if noise:
        return add_noise_and_distortion(img), background_color, text_color, font_name
    return img, background_color, text_color, font_name
