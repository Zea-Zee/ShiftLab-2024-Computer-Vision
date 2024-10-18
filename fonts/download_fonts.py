import os
import requests


custom_fonts = [
    "Arial",
    "Verdana",
    "Times New Roman",
    "Comic Sans MS",
    "Roboto",
    "Courier New",
    "Georgia",
    "Impact",
    "Calibri",
    "Tahoma",
    "Ubuntu",
    "Open Sans",
    "Lobster",
    "Montserrat",
    "Source Code Pro"
]


if not os.path.exists('fonts'):
    os.makedirs('fonts')


def download_font(font_name):
    font_url = f"https://fonts.googleapis.com/css2?family={font_name.replace(' ', '+')}:wght@400&display=swap"
    response = requests.get(font_url)

    if response.status_code == 200:
        css_lines = response.text.splitlines()
        for line in css_lines:
            if "url(" in line:
                start = line.find("url(") + 4
                end = line.find(")", start)
                font_file_url = line[start:end].replace('"', '').replace("'", "")
                font_response = requests.get(font_file_url)

                font_file_path = os.path.join('fonts', font_name.replace(' ', '_') + ".ttf")
                with open(font_file_path, 'wb') as f:
                    f.write(font_response.content)
                print(f"Скачан шрифт: {font_name}")
                return
    print(f"Не удалось скачать шрифт: {font_name}")

for font in custom_fonts:
    download_font(font)
