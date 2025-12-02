# 🎯 Eiquetas - Detección de Rostros

Aplicación Python para detectar rostros de **personas** y **animales** en imágenes usando modelos de deep learning de alta precisión.

## 🚀 Características

- ✅ **Detección de rostros humanos** con MTCNN (precisión ~95%)
- ✅ **Detección de animales** con YOLOv8 Medium (10 especies)
- ✅ **Procesamiento por lotes** de carpetas completas
- ✅ **Visualización** con bounding boxes de colores
- ✅ **Estadísticas** detalladas por imagen

## 📋 Requisitos

- Python 3.8 o superior
- ~60 MB de espacio para el modelo YOLOv8 Medium

## 🔧 Instalación

```bash
# 1. Clonar o descargar el proyecto
cd Eiquetas

# 2. Instalar dependencias
pip install -r requirements.txt
```

El modelo YOLOv8 Medium se descargará automáticamente en la primera ejecución (~52 MB).

## 📖 Uso

### Procesar una carpeta de imágenes

```bash
python main.py --folder input/
```

### Procesar una imagen individual

```bash
python main.py --image input/foto.jpg
```

### Mostrar resultados en pantalla

```bash
python main.py --image input/foto.jpg --show
```

### Ajustar precisión

```bash
# Más estricto para humanos
python main.py --folder input/ --human-conf 0.95

# Más sensible para animales
python main.py --folder input/ --animal-conf 0.2

# Usar modelo más preciso (más lento)
python main.py --folder input/ --yolo-model l
```

## ⚙️ Parámetros

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--image` | Ruta a imagen individual | - |
| `--folder` | Ruta a carpeta con imágenes | - |
| `--output` | Carpeta de salida | `output/` |
| `--human-conf` | Umbral de confianza humanos (0-1) | `0.9` |
| `--animal-conf` | Umbral de confianza animales (0-1) | `0.3` |
| `--yolo-model` | Tamaño modelo YOLO (n/s/m/l/x) | `m` |
| `--show` | Mostrar imágenes procesadas | `False` |
| `--no-confidence` | Ocultar nivel de confianza | `False` |
| `--no-stats` | Ocultar estadísticas | `False` |

## 🧠 Modelos Utilizados

### MTCNN - Detección de Rostros Humanos
- **Precisión**: 95-98% en condiciones normales
- **Características**: Detecta puntos faciales (ojos, nariz, boca)
- **Velocidad**: Rápida (~0.1-0.3s por imagen)

### YOLOv8 Medium - Detección de Animales
- **Precisión**: 90-95%
- **Velocidad**: ~1-2 segundos por imagen (CPU)
- **Animales detectados**: Perro, Gato, Caballo, Pájaro, Oveja, Vaca, Elefante, Oso, Cebra, Jirafa

## 📊 Formato de Salida

Las imágenes procesadas incluyen:
- 🟢 **Bounding boxes verdes**: Personas
- 🔵 **Bounding boxes azules**: Animales
- **Etiquetas**: Nombre + nivel de confianza
- **Overlay de estadísticas**: Total de detecciones por tipo

Ejemplo de salida:
```
output/
├── foto1_anotada.jpg
├── foto2_anotada.jpg
└── ...
```

## 📁 Estructura del Proyecto

```
Eiquetas/
├── src/
│   ├── face_detector.py      # Detector MTCNN para humanos
│   ├── animal_detector.py    # Detector YOLOv8 para animales
│   ├── image_processor.py    # Pipeline de procesamiento
│   └── visualizer.py          # Visualización y anotación
├── input/                     # Coloca tus imágenes aquí
├── output/                    # Resultados (generado automáticamente)
├── main.py                    # Aplicación principal
├── requirements.txt           # Dependencias
└── README.md                  # Este archivo
```

## 🎯 Ejemplos

### Ejemplo 1: Foto familiar
```bash
python main.py --image familia.jpg --show
```
**Resultado**: Detecta todos los rostros humanos con alta precisión

### Ejemplo 2: Foto con mascotas
```bash
python main.py --image mascotas.jpg --show
```
**Resultado**: Detecta perros, gatos u otros animales

### Ejemplo 3: Procesamiento masivo
```bash
python main.py --folder vacaciones/
```
**Resultado**: Procesa todas las fotos y guarda resultados en `output/`

## 💡 Consejos para Mejor Precisión

1. **Imágenes de calidad**: Mejor iluminación = mejor detección
2. **Rostros visibles**: No muy pequeños (>50x50 píxeles)
3. **Ajustar umbrales**:
   - ↑ Aumentar para menos falsos positivos
   - ↓ Disminuir para detectar más rostros
4. **Modelo YOLO más grande**: Para mejor precisión usar `--yolo-model l` o `x`

## 🔍 Limitaciones

- El modelo COCO solo incluye **10 clases de animales**
- No detecta: cerdos, conejos, peces, reptiles, insectos
- Requiere rostros razonablemente visibles
- GPU recomendada para procesamiento rápido de muchas imágenes

## 🐛 Solución de Problemas

**Detección lenta**:
```bash
python main.py --image foto.jpg --yolo-model n --max-size 1280
```

**Muchos falsos positivos**:
```bash
python main.py --image foto.jpg --human-conf 0.95 --animal-conf 0.5
```

**No detecta algunos rostros**:
```bash
python main.py --image foto.jpg --human-conf 0.7 --animal-conf 0.2
```

## 📝 Dependencias Principales

- `opencv-python` - Procesamiento de imágenes
- `mtcnn` - Detección de rostros humanos
- `ultralytics` - YOLOv8 para detección de animales
- `torch` - PyTorch (backend de deep learning)
- `numpy` - Operaciones numéricas

## 📄 Licencia

MIT License - Libre para uso personal y comercial

## 👨‍💻 Autor
Sebas Dev - 
Desarrollado con ❤️ usando Python y Deep Learning
