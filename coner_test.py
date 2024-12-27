import cv2
import numpy as np

def yolo_to_points(cx, cy, w, h, image_width, image_height):
    """
    将YOLO格式的标注转换为四个角点坐标
    cx, cy: 中心点坐标 (在 [0, 1] 范围内)
    w, h: 车牌框的宽度和高度 (在 [0, 1] 范围内)
    image_width, image_height: 图像的实际宽度和高度
    """
    # 计算车牌框的左上角和右下角
    x1 = int((cx - w / 2) * image_width)
    y1 = int((cy - h / 2) * image_height)
    x2 = int((cx + w / 2) * image_width)
    y2 = int((cy + h / 2) * image_height)

    # 返回四个角点的坐标，按顺时针方向排列
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

def get_rotate_angle(plate_points):
    """
    计算车牌的旋转角度
    """
    # 计算车牌区域的两条对角线的角度
    p1, p2, p3, p4 = plate_points
    
    # 计算左上角到右下角的角度
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = np.arctan2(dy, dx) * 180 / np.pi  # 计算弧度并转换为角度
    return angle

def rotate_image(image, angle, center=None):
    """
    旋转图像以矫正车牌倾斜
    """
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 进行仿射变换（旋转图像）
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    return rotated

def correct_plate_orientation(image, plate_points):
    """
    基于车牌的四个角点进行图像矫正
    """
    # 计算旋转角度
    angle = get_rotate_angle(plate_points)

    # 如果旋转角度过小，不需要旋转
    if abs(angle) < 5:
        return image

    # 旋转图像矫正
    corrected_image = rotate_image(image, angle)
    return corrected_image

def correct_plate_using_yolo(image, yolo_annotation, image_width, image_height):
    """
    使用YOLO格式标注的车牌框计算角点并进行矫正
    """
    cx, cy, w, h = yolo_annotation  # YOLO格式标注
    plate_points = yolo_to_points(cx, cy, w, h, image_width, image_height)
    print("车牌角点坐标：", plate_points)

    # 进行车牌矫正
    corrected_image = correct_plate_orientation(image, plate_points)

    return corrected_image

if __name__ == "__main__":
    img_path = '/data/data2/jx/car_plate_train/car_plate_rec/crop_20241125203018.png'  # 这里是输入图片路径
    image = cv2.imread(img_path)

    # YOLO标注的车牌信息（假设标注格式为(cx, cy, w, h)）
    # 标注的值是归一化的：cx, cy, w, h 在[0, 1]范围内
    yolo_annotation = (0.5, 0.5, 0.5, 0.5)  # 示例标注 (cx, cy, w, h)

    # 图像的宽度和高度
    image_width, image_height = image.shape[1], image.shape[0]

    # 进行车牌矫正
    corrected_image = correct_plate_using_yolo(image, yolo_annotation, image_width, image_height)

    # 保存矫正后的图像到本地
    output_path = 'corrected_plate.png'
    cv2.imwrite(output_path, corrected_image)

    print(f"矫正后的车牌图像已保存到：{output_path}")
