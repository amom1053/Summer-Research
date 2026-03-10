import tkinter as tk
from tkinter import filedialog
import pandas as pd
import cv2
import os
from PIL import Image

def read_csv_file():
    tk.Tk().withdraw()
    filename = filedialog.askopenfilename(title="select a file",
                               filetypes=[("CSV Files", "*.csv")])
    if filename:
        data = pd.read_csv(filename, header=None)
        return data
    else:
        return None

def save_csv_file(data):
    if data == None:
        print("save_csv_file: The input data is blank.")
        return
    tk.Tk().withdraw()
    filename = filedialog.asksaveasfilename(title="select a file",
                               filetypes=[("CSV Files", "*.csv")])
    if filename:
        _, ext = os.path.splitext(filename)
        if not ext:
            ext = ".csv"  # default is BMP format if no extension is provided
            filename += ext
        df = pd.DataFrame(data)
        df.to_csv(filename)
        print("CSV file saved successfully!")
    else:
        print("CSV file save operation cancelled.")

def read_image_file():
    tk.Tk().withdraw()
    filename = filedialog.askopenfilename(title="select a file",
                               filetypes=[("bmp image", "*.bmp"),
                                          ("all files", "*.*")])
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    return img

def save_image_file(data):
    if data is None:
        print("save_image_file: The input data is blank.")
        return
    tk.Tk().withdraw()
    filename = filedialog.asksaveasfilename(title="select a file",
                                filetypes=[("bmp images", "*.bmp")])
    if filename:
        _, ext = os.path.splitext(filename)
        if not ext:
            ext = ".bmp" # default is BMP format if no extension is provided
            filename += ext
        cv2.imwrite(filename, data)
        print("Image saved successfully!")
    else:
        print("Image save operation cancelled.")

def read_image_folder():
    tk.Tk().withdraw()
    folder_path = filedialog.askdirectory(title="Select a folder")
    if folder_path:
        image_list = []
        image_names = []
        for entry in os.scandir(folder_path):
            if entry.is_file():
                _, ext = os.path.splitext(entry.name)
                if ext.lower() in ('jpg', 'jpeg', '.png', '.bmp'):
                    try:
                        image = cv2.imread(entry.path, cv2.IMREAD_GRAYSCALE)
                        image_list.append(image)
                        image_names.append(entry.name)
                    except:
                        print(f"Failed to read image: {entry.name}")
        print("Number of loaded images: ", len(image_list))
        return image_list, image_names
    else:
        print("No folder selected.")
        return [], []


def save_images_to_folder(images, names):
    # Open file dialog to choose the output folder
    tk.Tk().withdraw()
    output_folder = filedialog.askdirectory(title="Select Output Folder")

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for name, image in zip(names, images):
        # Construct the output file path
        output_path = os.path.join(output_folder, name + "_processed.bmp")

        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Save the image
        image.save(output_path)

        print(f"Processed image saved as: {output_path}")

