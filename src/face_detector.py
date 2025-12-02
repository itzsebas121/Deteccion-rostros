"""
Detector de rostros humanos usando MTCNN
"""

import cv2
import numpy as np
from mtcnn import MTCNN
from typing import List, Dict, Tuple


class HumanFaceDetector:
    """
    Detector de rostros humanos de alta precisión usando MTCNN
    (Multi-task Cascaded Convolutional Networks)
    """
    
    def __init__(self, confidence_threshold: float = 0.9):
        """
        Inicializa el detector de rostros humanos
        
        Args:
            confidence_threshold: Umbral mínimo de confianza para considerar una detección válida
        """
        self.confidence_threshold = confidence_threshold
        self.detector = MTCNN()
        print(f"✓ HumanFaceDetector inicializado (umbral: {confidence_threshold})")
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detecta rostros humanos en una imagen
        
        Args:
            image: Imagen en formato numpy array (BGR)
        
        Returns:
            Lista de diccionarios con información de cada rostro detectado:
            - bbox: [x, y, width, height]
            - confidence: nivel de confianza (0-1)
            - keypoints: puntos faciales (ojos, nariz, boca)
            - label: 'Persona'
        """
        # Convertir de BGR a RGB (MTCNN espera RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detectar rostros
        detections = self.detector.detect_faces(rgb_image)
        
        # Filtrar por umbral de confianza
        results = []
        for detection in detections:
            confidence = detection['confidence']
            
            if confidence >= self.confidence_threshold:
                bbox = detection['box']  # [x, y, width, height]
                keypoints = detection['keypoints']
                
                results.append({
                    'bbox': bbox,
                    'confidence': float(confidence),
                    'keypoints': keypoints,
                    'label': 'Persona',
                    'type': 'human'
                })
        
        return results
    
    def get_face_count(self, image: np.ndarray) -> int:
        """
        Cuenta el número de rostros humanos en una imagen
        
        Args:
            image: Imagen en formato numpy array
        
        Returns:
            Número de rostros detectados
        """
        detections = self.detect(image)
        return len(detections)
