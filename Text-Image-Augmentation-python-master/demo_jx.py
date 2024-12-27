import cv2
import numpy as np
import os
from augment import distort, stretch, perspective
import argparse
import random

# 新增红外效果
def simulate_infrared_effect(image):
    """Simulate infrared camera effect with natural gray levels for the license plate."""
    print("[Debug] Simulating infrared effect...")

    # 转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 自适应阈值分割，生成带有灰度层次的高亮背景
    adaptive_binary = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10
    )

    # 增强亮度和对比度
    enhanced_image = cv2.addWeighted(adaptive_binary, 1.5, gray_image, -0.5, 50)

    # 转换回 3 通道以便与原始图像格式一致
    infrared_effect_image = cv2.merge([enhanced_image, enhanced_image, enhanced_image])

    print("[Debug] Infrared effect simulation completed.")
    return infrared_effect_image
# 调整亮度与对比度
def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    print(f"[Debug] Adjusting brightness/contrast with alpha={alpha}, beta={beta}")
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


# 添加高斯噪声
def add_gaussian_noise(image, mean=0.01, std=0.001):
    """Add Gaussian noise to the image."""
    print(f"[Debug] Adding Gaussian noise with mean={mean}, std={std}")
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    return cv2.add(image, noise)


# 旋转图片
def rotate_image(image, angle):
    """Rotate the image by a specific angle."""
    try:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        print(f"[Debug] Rotating image by angle: {angle}")
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))
    except Exception as e:
        print(f"[Error] Exception during rotation: {e}")
        return image


# 模拟反光
def simulate_glare(image, center=None, radius=30, intensity=100):
    h, w = image.shape[:2]
    if w < radius * 2 or h < radius * 2:
        print(f"[Warning] Image too small for glare simulation: w={w}, h={h}, radius={radius}")
        return image
    if center is None:
        center = (safe_random_int(radius, w - radius), safe_random_int(radius, h - radius))
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.circle(mask, center, radius, (intensity, intensity, intensity), -1)
    return cv2.addWeighted(image, 2.5, mask, 3, 0)


def safe_random_int(low, high):
    if low >= high:
        print(f"[Error] Invalid range for random.randint: low={low}, high={high}")
        return low  # 或者返回默认值
    return np.random.randint(low, high)



# 遍历所有文件
def allFileList(rootfile, allFile):
    folder = os.listdir(rootfile)
    for temp in folder:
        fileName = os.path.join(rootfile, temp)
        if os.path.isfile(fileName):
            allFile.append(fileName)
        else:
            allFileList(fileName, allFile)


# 加载图片并调试
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


# 处理图片
def process_image(im, temp, saveFile):
    try:
        if len(im.shape) == 3:
            _, _, c = im.shape
        else:
            c = 1  # Grayscale image

        if c == 4:
            print(f"[Warning] Image has 4 channels (RGBA), converting to RGB: {temp}")
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        elif c != 3:
            print(f"[Warning] Image is not RGB, converting: {temp}")
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

        # 原始增强方法
        print(f"[Debug] Applying distortion...")
        distort_img = distort(im, 8)
        print(f"[Debug] Distortion applied successfully.")

        print(f"[Debug] Applying stretching...")
        stretch_img = stretch(distort_img, 8)
        print(f"[Debug] Stretching applied successfully.")

        print(f"[Debug] Applying perspective transformation...")
        perspective_img = perspective(stretch_img)
        print(f"[Debug] Perspective transformation applied successfully.")

        # 新增增强方法
        enhanced_images = [
            adjust_brightness_contrast(perspective_img, alpha=random.uniform(0.6, 1.4), beta=random.randint(-30, 30)),
            add_gaussian_noise(perspective_img, mean=0, std=0.62),  # 调整噪声标准差
            rotate_image(perspective_img, angle=random.uniform(-20, 20)),
            simulate_glare(perspective_img, radius=5, intensity=300),
            simulate_infrared_effect(im)  # 新增的红外效果
        ]

        # 保存增强后的图片
        if not os.path.exists(saveFile):
            os.makedirs(saveFile)

        for i, img in enumerate(enhanced_images):
            newPath = os.path.join(saveFile, f"{os.path.splitext(os.path.basename(temp))[0]}_aug_{i}.jpg")
            print(f"[Info] Saving augmented image to: {newPath}")
            cv2.imwrite(newPath, img)

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

        img = debug_image_loading(temp)
        if img is None:
            continue

        process_image(img, temp, saveFile)
