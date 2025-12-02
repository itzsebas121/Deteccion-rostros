"""
Procesador de imágenes que coordina la detección de rostros humanos y animales
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from .face_detector import HumanFaceDetector
from .animal_detector import AnimalDetector


class ImageProcessor:
    """
    Pipeline de procesamiento de imágenes que coordina
    la detección de rostros humanos y animales
    """
    
    def __init__(
        self,
        human_confidence: float = 0.9,
        animal_confidence: float = 0.3,
        yolo_model_size: str = 'm',
        max_image_size: int = 1920
    ):
        """
        Inicializa el procesador de imágenes
        
        Args:
            human_confidence: Umbral de confianza para detección de humanos (default: 0.9)
            animal_confidence: Umbral de confianza para detección de animales (default: 0.3)
            yolo_model_size: Tamaño del modelo YOLO (default: 'm' - medium, más preciso)
            max_image_size: Tamaño máximo de la imagen (se redimensiona si es mayor)
        """
        self.max_image_size = max_image_size
        
        print("Inicializando detectores...")
        self.human_detector = HumanFaceDetector(confidence_threshold=human_confidence)
        self.animal_detector = AnimalDetector(
            confidence_threshold=animal_confidence,
            model_size=yolo_model_size
        )
        print("✓ ImageProcessor listo\n")
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Carga una imagen desde un archivo
        
        Args:
            image_path: Ruta al archivo de imagen
        
        Returns:
            Imagen en formato numpy array o None si hay error
        """
        try:
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"✗ Error: No se pudo cargar la imagen {image_path}")
                return None
            
            # Redimensionar si es muy grande
            height, width = image.shape[:2]
            if max(height, width) > self.max_image_size:
                scale = self.max_image_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                print(f"  → Imagen redimensionada de {width}x{height} a {new_width}x{new_height}")
            
            return image
            
        except Exception as e:
            print(f"✗ Error al cargar imagen: {e}")
            return None
    
    def process_image(self, image: np.ndarray) -> Dict:
        """
        Procesa una imagen detectando rostros humanos y animales
        
        Args:
            image: Imagen en formato numpy array
        
        Returns:
            Diccionario con resultados:
            - humans: lista de detecciones humanas
            - animals: lista de detecciones de animales
            - total_detections: número total de detecciones
            - stats: estadísticas del procesamiento
        """
        # Detectar humanos
        human_detections = self.human_detector.detect(image)
        
        # Detectar animales
        animal_detections = self.animal_detector.detect(image)
        
        # Combinar resultados
        all_detections = human_detections + animal_detections
        
        # Calcular estadísticas
        stats = {
            'total_humans': len(human_detections),
            'total_animals': len(animal_detections),
            'total_detections': len(all_detections),
            'image_shape': image.shape
        }
        
        # Agregar conteo por tipo de animal
        if animal_detections:
            animal_counts = {}
            for detection in animal_detections:
                label = detection['label']
                animal_counts[label] = animal_counts.get(label, 0) + 1
            stats['animals_by_type'] = animal_counts
        
        return {
            'humans': human_detections,
            'animals': animal_detections,
            'all_detections': all_detections,
            'stats': stats
        }
    
    def process_image_file(self, image_path: str) -> Optional[Dict]:
        """
        Procesa un archivo de imagen
        
        Args:
            image_path: Ruta al archivo de imagen
        
        Returns:
            Diccionario con resultados o None si hay error
        """
        image = self.load_image(image_path)
        
        if image is None:
            return None
        
        results = self.process_image(image)
        results['image'] = image
        results['image_path'] = image_path
        
        return results
    
    def process_folder(self, folder_path: str, extensions: List[str] = None) -> List[Dict]:
        """
        Procesa todas las imágenes en una carpeta
        
        Args:
            folder_path: Ruta a la carpeta
            extensions: Lista de extensiones permitidas (default: jpg, jpeg, png, webp)
        
        Returns:
            Lista de resultados para cada imagen
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"✗ Error: La carpeta {folder_path} no existe")
            return []
        
        # Buscar imágenes
        image_files = []
        for ext in extensions:
            image_files.extend(folder.glob(f'*{ext}'))
            image_files.extend(folder.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"✗ No se encontraron imágenes en {folder_path}")
            return []
        
        print(f"Procesando {len(image_files)} imágenes...\n")
        
        results = []
        for i, image_file in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] Procesando: {image_file.name}")
            result = self.process_image_file(str(image_file))
            
            if result:
                results.append(result)
                stats = result['stats']
                print(f"  ✓ Detectados: {stats['total_humans']} personas, {stats['total_animals']} animales")
            
            print()
        
        return results
