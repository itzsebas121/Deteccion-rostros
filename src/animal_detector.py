"""
Detector de animales usando YOLOv8
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict


class AnimalDetector:
    """
    Detector de animales usando YOLOv8
    Detecta animales usando el dataset COCO completo
    """
    
    # TODAS las clases de animales en COCO dataset (80 clases totales)
    ANIMAL_CLASSES = {
        14: 'bird',
        15: 'cat',
        16: 'dog',
        17: 'horse',
        18: 'sheep',
        19: 'cow',
        20: 'elephant',
        21: 'bear',
        22: 'zebra',
        23: 'giraffe'
    }
    
    # Mapeo a español
    SPANISH_LABELS = {
        'bird': 'Pájaro',
        'cat': 'Gato',
        'dog': 'Perro',
        'horse': 'Caballo',
        'sheep': 'Oveja',
        'cow': 'Vaca',
        'elephant': 'Elefante',
        'bear': 'Oso',
        'zebra': 'Cebra',
        'giraffe': 'Jirafa'
    }
    
    def __init__(self, confidence_threshold: float = 0.3, model_size: str = 'm'):
        """
        Inicializa el detector de animales
        
        Args:
            confidence_threshold: Umbral mínimo de confianza (default: 0.3 para mejor detección)
            model_size: Tamaño del modelo YOLOv8 ('n', 's', 'm', 'l', 'x')
                       n = nano (más rápido, menos preciso)
                       m = medium (balance, RECOMENDADO)
                       x = extra large (más lento, más preciso)
        """
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(f'yolov8{model_size}.pt')
        print(f"✓ AnimalDetector inicializado (modelo: yolov8{model_size}, umbral: {confidence_threshold})")
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detecta animales en una imagen
        
        Args:
            image: Imagen en formato numpy array (BGR)
        
        Returns:
            Lista de diccionarios con información de cada animal detectado:
            - bbox: [x, y, width, height]
            - confidence: nivel de confianza (0-1)
            - label: nombre del animal en español
            - class_name: nombre de la clase en inglés
            - type: 'animal'
        """
        # Realizar detección
        results = self.model(image, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Filtrar solo animales con confianza suficiente
            if class_id in self.ANIMAL_CLASSES and confidence >= self.confidence_threshold:
                # Obtener coordenadas del bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Convertir a formato [x, y, width, height]
                bbox = [
                    int(x1),
                    int(y1),
                    int(x2 - x1),
                    int(y2 - y1)
                ]
                
                class_name = self.ANIMAL_CLASSES[class_id]
                spanish_label = self.SPANISH_LABELS[class_name]
                
                detections.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'label': spanish_label,
                    'class_name': class_name,
                    'type': 'animal'
                })
        
        return detections
    
    def get_animal_count(self, image: np.ndarray) -> int:
        """
        Cuenta el número de animales en una imagen
        
        Args:
            image: Imagen en formato numpy array
        
        Returns:
            Número de animales detectados
        """
        detections = self.detect(image)
        return len(detections)
    
    def get_animals_by_type(self, image: np.ndarray) -> Dict[str, int]:
        """
        Cuenta animales agrupados por tipo
        
        Args:
            image: Imagen en formato numpy array
        
        Returns:
            Diccionario con conteo por tipo de animal
        """
        detections = self.detect(image)
        counts = {}
        
        for detection in detections:
            label = detection['label']
            counts[label] = counts.get(label, 0) + 1
        
        return counts
