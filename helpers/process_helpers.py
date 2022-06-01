import operator
import cv2
import numpy as np


def find_extreme_corners(polygon, limit_fn, compare_fn):
    # Limite de la funcion minima y maxima de la funcionlimit_fn is the min or max function
    # Compara compare_fn si en np.add o np.subtract se obtendra la función

    # Si intentamos encontrar los valores de las esquinas se hará por las coordenadas (x-y) más pequeños
    section, _ = limit_fn(enumerate([compare_fn(pt[0][0], pt[0][1]) for pt in polygon]),
                          key=operator.itemgetter(1))

    return polygon[section][0][0], polygon[section][0][1]


def draw_extreme_corners(pts, original):
    cv2.circle(original, pts, 7, (0, 255, 0), cv2.FILLED)


def clean_helper(img):
    # Realiza la función print(np.isclose(img, 0).sum())
    if np.isclose(img, 0).sum() / (img.shape[0] * img.shape[1]) >= 0.95:
        return np.zeros_like(img), False

    # Verifica si se tienen imagenes en blanco por lo que no se analizara y no se realizará nada
    height, width = img.shape
    mid = width // 2
    if np.isclose(img[:, int(mid - width * 0.4):int(mid + width * 0.4)], 0).sum() / (2 * width * 0.4 * height) >= 0.90:
        return np.zeros_like(img), False

    # Centro de la imagen
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[0])

    start_x = (width - w) // 2
    start_y = (height - h) // 2
    new_img = np.zeros_like(img)
    new_img[start_y:start_y + h, start_x:start_x + w] = img[y:y + h, x:x + w]

    return new_img, True


def grid_line_helper(img, shape_location, length=10):
    clone = img.copy()
    # Si son lineas horizonatales las ubica por shape_location 1, para verticales son 0
    row_or_col = clone.shape[shape_location]
    # Encuentra la distancia entre las lineas
    size = row_or_col // length

    # Encuentra el kernel
    if shape_location == 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))

    # Dilata las lineas
    clone = cv2.erode(clone, kernel)
    clone = cv2.dilate(clone, kernel)

    return clone


def draw_lines(img, lines):
    
    clone = img.copy()
    lines = np.squeeze(lines)

    for rho, theta in lines:
        # Busca que ten estiradas estan las lineas
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(clone, (x1, y1), (x2, y2), (255, 255, 255), 4)
    return clone
