import cv2
import numpy as np
import os
from augment import distort, stretch, perspective
import argparse

def allFileList(rootfile, allFile):
    folder = os.listdir(rootfile)
    for temp in folder:
        fileName = os.path.join(rootfile, temp)
        if os.path.isfile(fileName):
            allFile.append(fileName)
        else:
            allFileList(fileName, allFile)

def create_gif(image_list, gif_name, duration=0.1):
    frames = []
    for image in image_list:
        frames.append(image)
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

def debug_image_loading(img_path):
    try:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        if img is None:
            print(f"[Error] Unable to load image: {img_path}")
            return None
        else:
            print(f"[Info] Image loaded successfully: {img_path}")
            return img
    except Exception as e:
        print(f"[Error] Exception while loading image {img_path}: {e}")
        return None

def process_image(im, temp, saveFile):
    try:
        # Check the number of channels in the image
        if len(im.shape) == 3:
            _, _, c = im.shape
        else:
            c = 1  # Grayscale image

        # Check if image has 4 channels (RGBA)
        if c == 4:
            print(f"[Warning] Image has 4 channels (RGBA), converting to RGB: {temp}")
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        elif c != 3:
            print(f"[Warning] Image is not RGB, converting: {temp}")
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

        distort_img_list = list()
        stretch_img_list = list()
        perspective_img_list = list()

        # Applying distortion, stretching and perspective transformations
        distort_img = distort(im, 8)
        distort_img_list.append(distort_img)
        stretch_img = stretch(distort_img, 8)
        stretch_img_list.append(stretch_img)
        perspective_img = perspective(stretch_img)
        perspective_img_list.append(perspective_img)

        # Ensure the save directory exists
        if not os.path.exists(saveFile):
            os.makedirs(saveFile)

        # Saving the transformed image with the same file name
        newPath = os.path.join(saveFile, os.path.basename(temp))
        
        print(f"[Info] Saving image to: {newPath}")

        # Try using imwrite directly to save the image
        success = cv2.imwrite(newPath, perspective_img)

        if success:
            print(f"[Info] Processed and saved image: {newPath}")
        else:
            print(f"[Error] Failed to save image: {newPath}")

    except Exception as e:
        print(f"[Error] Exception during transformation or saving image {temp}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='/mnt/jx/data/test', help='Source directory of images')
    parser.add_argument('--dst_path', type=str, default='/mnt/jx/data/aug', help='Destination directory to save transformed images')
    opt = parser.parse_args()
    rootFile = opt.src_path
    saveFile = opt.dst_path

    fileList = []
    allFileList(rootFile, fileList)

    picCount = 0
    for temp in fileList:
        print(f"[Info] Processing image {picCount}: {temp}")
        picCount += 1

        # Load the image and process it
        img = debug_image_loading(temp)
        if img is None:
            continue  # Skip this image if it couldn't be loaded

        # Process the loaded image and save it with the same file name
        process_image(img, temp, saveFile)
