"""
Visualizador y anotador de imágenes con detecciones
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple


class Visualizer:
    """
    Visualiza y anota imágenes con las detecciones de rostros y animales
    """
    
    # Colores para diferentes tipos de detecciones (BGR)
    COLORS = {
        'human': (0, 255, 0),      # Verde para personas
        'animal': (255, 100, 0)     # Azul-cyan para animales
    }
    
    def __init__(
        self,
        font_scale: float = 0.6,
        thickness: int = 2,
        box_thickness: int = 2
    ):
        """
        Inicializa el visualizador
        
        Args:
            font_scale: Escala de la fuente para las etiquetas
            thickness: Grosor del texto
            box_thickness: Grosor de los bounding boxes
        """
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness = thickness
        self.box_thickness = box_thickness
    
    def draw_detection(
        self,
        image: np.ndarray,
        detection: Dict,
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Dibuja una detección en la imagen
        
        Args:
            image: Imagen donde dibujar
            detection: Diccionario con información de la detección
            show_confidence: Si mostrar el nivel de confianza
        
        Returns:
            Imagen con la detección dibujada
        """
        bbox = detection['bbox']
        label = detection['label']
        confidence = detection['confidence']
        detection_type = detection['type']
        
        # Obtener color según el tipo
        color = self.COLORS.get(detection_type, (255, 255, 255))
        
        # Extraer coordenadas
        x, y, w, h = bbox
        
        # Dibujar bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, self.box_thickness)
        
        # Preparar texto
        if show_confidence:
            text = f"{label} ({confidence:.2f})"
        else:
            text = label
        
        # Calcular tamaño del texto para el fondo
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.thickness
        )
        
        # Dibujar fondo del texto
        cv2.rectangle(
            image,
            (x, y - text_height - baseline - 5),
            (x + text_width + 5, y),
            color,
            -1  # Relleno
        )
        
        # Dibujar texto
        cv2.putText(
            image,
            text,
            (x + 2, y - baseline - 2),
            self.font,
            self.font_scale,
            (255, 255, 255),  # Blanco
            self.thickness
        )
        
        return image
    
    def draw_all_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Dibuja todas las detecciones en la imagen
        
        Args:
            image: Imagen donde dibujar
            detections: Lista de detecciones
            show_confidence: Si mostrar el nivel de confianza
        
        Returns:
            Imagen con todas las detecciones dibujadas
        """
        # Hacer una copia para no modificar la original
        annotated_image = image.copy()
        
        for detection in detections:
            annotated_image = self.draw_detection(
                annotated_image,
                detection,
                show_confidence
            )
        
        return annotated_image
    
    def add_stats_overlay(
        self,
        image: np.ndarray,
        stats: Dict,
        position: str = 'top-left'
    ) -> np.ndarray:
        """
        Agrega estadísticas en la imagen
        
        Args:
            image: Imagen donde agregar las estadísticas
            stats: Diccionario con estadísticas
            position: Posición del overlay ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        
        Returns:
            Imagen con estadísticas
        """
        # Preparar texto
        lines = [
            f"Personas: {stats['total_humans']}",
            f"Animales: {stats['total_animals']}",
            f"Total: {stats['total_detections']}"
        ]
        
        # Agregar desglose de animales si existe
        if 'animals_by_type' in stats:
            lines.append("---")
            for animal_type, count in stats['animals_by_type'].items():
                lines.append(f"{animal_type}: {count}")
        
        # Calcular dimensiones del overlay
        padding = 10
        line_height = 25
        max_width = max([cv2.getTextSize(line, self.font, self.font_scale, self.thickness)[0][0] for line in lines])
        overlay_width = max_width + 2 * padding
        overlay_height = len(lines) * line_height + 2 * padding
        
        # Calcular posición
        h, w = image.shape[:2]
        if position == 'top-left':
            x, y = 10, 10
        elif position == 'top-right':
            x, y = w - overlay_width - 10, 10
        elif position == 'bottom-left':
            x, y = 10, h - overlay_height - 10
        else:  # bottom-right
            x, y = w - overlay_width - 10, h - overlay_height - 10
        
        # Dibujar fondo semi-transparente
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (x, y),
            (x + overlay_width, y + overlay_height),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Dibujar texto
        for i, line in enumerate(lines):
            text_y = y + padding + (i + 1) * line_height - 5
            cv2.putText(
                image,
                line,
                (x + padding, text_y),
                self.font,
                self.font_scale,
                (255, 255, 255),
                self.thickness
            )
        
        return image
    
    def save_annotated_image(
        self,
        image: np.ndarray,
        detections: List[Dict],
        output_path: str,
        stats: Dict = None,
        show_confidence: bool = True,
        show_stats: bool = True
    ) -> bool:
        """
        Guarda una imagen anotada con las detecciones
        
        Args:
            image: Imagen original
            detections: Lista de detecciones
            output_path: Ruta donde guardar la imagen
            stats: Estadísticas a mostrar
            show_confidence: Si mostrar confianza en las etiquetas
            show_stats: Si mostrar overlay de estadísticas
        
        Returns:
            True si se guardó correctamente
        """
        try:
            # Dibujar detecciones
            annotated = self.draw_all_detections(image, detections, show_confidence)
            
            # Agregar estadísticas si se solicita
            if show_stats and stats:
                annotated = self.add_stats_overlay(annotated, stats, 'top-left')
            
            # Crear directorio si no existe
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Guardar imagen
            cv2.imwrite(output_path, annotated)
            return True
            
        except Exception as e:
            print(f"✗ Error al guardar imagen: {e}")
            return False
    
    def _convert_to_serializable(self, obj):
        """
        Convierte tipos de NumPy a tipos nativos de Python para serialización JSON
        
        Args:
            obj: Objeto a convertir
        
        Returns:
            Objeto convertido a tipo serializable
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_serializable(item) for item in obj)
        else:
            return obj
    
    def save_detections_json(
        self,
        detections: List[Dict],
        output_path: str,
        stats: Dict = None,
        image_path: str = None
    ) -> bool:
        """
        Guarda las detecciones en formato JSON
        
        Args:
            detections: Lista de detecciones
            output_path: Ruta donde guardar el JSON
            stats: Estadísticas adicionales
            image_path: Ruta de la imagen original
        
        Returns:
            True si se guardó correctamente
        """
        try:
            # Preparar datos y convertir tipos NumPy a tipos nativos de Python
            data = {
                'image_path': image_path,
                'detections': self._convert_to_serializable(detections),
                'stats': self._convert_to_serializable(stats)
            }
            
            # Crear directorio si no existe
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Guardar JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"✗ Error al guardar JSON: {e}")
            return False
    
    def show_image(self, image: np.ndarray, window_name: str = "Detecciones") -> None:
        """
        Muestra una imagen en una ventana
        
        Args:
            image: Imagen a mostrar
            window_name: Nombre de la ventana
        """
        cv2.imshow(window_name, image)
        print(f"\nPresiona cualquier tecla para cerrar la ventana...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
