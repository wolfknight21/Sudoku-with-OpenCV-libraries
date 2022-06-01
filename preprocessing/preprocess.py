import cv2


def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Difimuna
    blur = cv2.GaussianBlur(img_gray, (9, 9), 0)

    # Lo limita
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Invierte para que las líneas de la cuadrícula y el texto sean blancos
    inverted = cv2.bitwise_not(thresh, 0)

    # Obtiene un kernel rectángulo
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # Transforma para eliminar algo de ruido como puntos aleatorios
    morph = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)

    # Dilata e incrementa los lados de los bordes
    result = cv2.dilate(morph, kernel, iterations=1)
    return result
