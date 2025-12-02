"""
Eiquetas - Aplicación de Detección de Rostros
Detecta rostros de personas y animales en imágenes
"""

import argparse
import time
from pathlib import Path
from src.image_processor import ImageProcessor
from src.visualizer import Visualizer


def main():
    parser = argparse.ArgumentParser(
        description='Detecta rostros de personas y animales en imágenes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Procesar una imagen
  python main.py --image input/foto.jpg
  
  # Procesar una carpeta
  python main.py --folder input/
  
  # Ajustar umbrales de confianza
  python main.py --image foto.jpg --human-conf 0.95 --animal-conf 0.6
  
  # Usar modelo YOLO más preciso (más lento)
  python main.py --image foto.jpg --yolo-model m
  
  # Mostrar imágenes procesadas
  python main.py --image foto.jpg --show
        """
    )
    
    # Argumentos de entrada
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image',
        type=str,
        help='Ruta a una imagen individual'
    )
    input_group.add_argument(
        '--folder',
        type=str,
        help='Ruta a una carpeta con imágenes'
    )
    
    # Argumentos de salida
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Carpeta de salida para imágenes anotadas (default: output/)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Mostrar imágenes procesadas en ventana'
    )
    
    # Argumentos de configuración
    parser.add_argument(
        '--human-conf',
        type=float,
        default=0.9,
        help='Umbral de confianza para detección de humanos (0-1, default: 0.9)'
    )
    parser.add_argument(
        '--animal-conf',
        type=float,
        default=0.3,
        help='Umbral de confianza para detección de animales (0-1, default: 0.3 - más sensible)'
    )
    parser.add_argument(
        '--yolo-model',
        type=str,
        choices=['n', 's', 'm', 'l', 'x'],
        default='m',
        help='Tamaño del modelo YOLO (n=nano rápido, m=medium RECOMENDADO, x=extra large preciso, default: m)'
    )
    parser.add_argument(
        '--max-size',
        type=int,
        default=1920,
        help='Tamaño máximo de imagen (se redimensiona si es mayor, default: 1920)'
    )
    parser.add_argument(
        '--no-confidence',
        action='store_true',
        help='No mostrar nivel de confianza en las etiquetas'
    )
    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='No mostrar overlay de estadísticas en las imágenes'
    )
    
    args = parser.parse_args()
    
    # Banner
    print("=" * 60)
    print("  EIQUETAS - Detección de Rostros")
    print("  Personas y Animales")
    print("=" * 60)
    print()
    
    # Inicializar procesador
    processor = ImageProcessor(
        human_confidence=args.human_conf,
        animal_confidence=args.animal_conf,
        yolo_model_size=args.yolo_model,
        max_image_size=args.max_size
    )
    
    # Inicializar visualizador
    visualizer = Visualizer()
    
    # Procesar imágenes
    start_time = time.time()
    
    if args.image:
        # Procesar imagen individual
        print(f"Procesando imagen: {args.image}\n")
        results = [processor.process_image_file(args.image)]
    else:
        # Procesar carpeta
        results = processor.process_folder(args.folder)
    
    # Filtrar resultados None
    results = [r for r in results if r is not None]
    
    if not results:
        print("✗ No se procesaron imágenes")
        return
    
    # Guardar resultados
    print("\n" + "=" * 60)
    print("Guardando resultados...")
    print("=" * 60 + "\n")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_humans = 0
    total_animals = 0
    
    for result in results:
        image_path = Path(result['image_path'])
        image_name = image_path.stem
        
        # Actualizar contadores
        total_humans += result['stats']['total_humans']
        total_animals += result['stats']['total_animals']
        
        # Guardar imagen anotada
        output_image_path = output_dir / f"{image_name}_anotada.jpg"
        success = visualizer.save_annotated_image(
            result['image'],
            result['all_detections'],
            str(output_image_path),
            result['stats'],
            show_confidence=not args.no_confidence,
            show_stats=not args.no_stats
        )
        if success:
            print(f"✓ Imagen guardada: {output_image_path}")
        
        # Mostrar imagen si se solicita
        if args.show:
            annotated = visualizer.draw_all_detections(
                result['image'],
                result['all_detections'],
                show_confidence=not args.no_confidence
            )
            if not args.no_stats:
                annotated = visualizer.add_stats_overlay(annotated, result['stats'])
            visualizer.show_image(annotated, f"Detecciones - {image_name}")
        
        print()
    
    # Estadísticas finales
    elapsed_time = time.time() - start_time
    
    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Imágenes procesadas: {len(results)}")
    print(f"Total personas detectadas: {total_humans}")
    print(f"Total animales detectados: {total_animals}")
    print(f"Tiempo total: {elapsed_time:.2f} segundos")
    print(f"Tiempo promedio por imagen: {elapsed_time/len(results):.2f} segundos")
    print("=" * 60)


if __name__ == '__main__':
    main()
