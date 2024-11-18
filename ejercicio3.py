import cv2
import numpy as np
import matplotlib.pyplot as plt
# Paso 1: Cargar y preprocesar la imagen
image = cv2.imread('circunsferencia3.jpg', cv2.IMREAD_GRAYSCALE)
blurred_image = cv2.medianBlur(image, 5)  # Suavizado para reducir el ruido

# Paso 2: Aplicación de la Transformada de Hough para Círculos
circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                           param1=50, param2=30, minRadius=0, maxRadius=0)

# Paso 3: Dibujar los círculos detectados
circle_image = np.copy(image)
if circles is not None:
    circles = np.uint16(np.around(circles))  # Redondeo de coordenadas
    for i in circles[0, :]:
        center = (i[0], i[1])  # Centro del círculo
        radius = i[2]  # Radio del círculo
        # Dibujar el centro del círculo
        cv2.circle(circle_image, center, 1, (255, 0, 0), 3)
        # Dibujar el contorno del círculo
        cv2.circle(circle_image, center, radius, (0, 255, 0), 3)

# Mostrar resultados
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(image, cmap="gray")
plt.subplot(1, 2, 2)
plt.title("Círculos Detectados")
plt.imshow(circle_image, cmap="gray")
plt.show()
