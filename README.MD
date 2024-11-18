# Detección de Círculos con la Transformada de Hough

Este proyecto utiliza la **Transformada de Hough** para detectar círculos en imágenes. El script está diseñado para cargar una imagen, aplicar técnicas de preprocesamiento, y finalmente detectar y visualizar los círculos identificados.

---

## **Descripción**

El script realiza los siguientes pasos:

1. **Cargar y preprocesar la imagen**:

   - La imagen se carga en escala de grises.
   - Se aplica un filtro de desenfoque para reducir el ruido.

2. **Aplicación de la Transformada de Hough para círculos**:

   - Se detectan círculos utilizando el método `cv2.HoughCircles`.

3. **Dibujo de los círculos detectados**:

   - Los círculos detectados se dibujan sobre la imagen, marcando tanto su contorno como su centro.

4. **Visualización de los resultados**:
   - Se muestran la imagen original y la imagen procesada con los círculos detectados, una al lado de la otra.

---

## **Requisitos**

Antes de ejecutar el código, asegúrate de tener instaladas las siguientes dependencias:

- **Python 3.7 o superior**
- **OpenCV**: Biblioteca para procesamiento de imágenes.
- **NumPy**: Para cálculos matriciales.
- **Matplotlib**: Para la visualización de imágenes.

### **Instalación de dependencias**

Ejecuta el siguiente comando para instalar las bibliotecas necesarias:

```bash
pip install opencv-python matplotlib numpy
```
