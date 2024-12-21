import cv2
import numpy as np
import re
from tqdm import tqdm

def parse_glp_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    scale_match = re.search(r'EQUIV\s+\d+\s+(\d+)\s+MICRON', content)
    if not scale_match:
        raise ValueError("无法找到比例信息")
    scale_factor = int(scale_match.group(1))
    print(f"Scale Factor: {scale_factor}")

    polygons = []
    polygon_pattern = re.compile(r'PGON\s+N\s+M1\s*([^\n]+)')
    lines = content.split('\n')
    for line in tqdm(lines, desc="Parsing Polygons"):
        match = polygon_pattern.search(line)
        if match:
            coords = list(map(int, match.group(1).split()))
            polygons.append([(coords[i], coords[i+1]) for i in range(0, len(coords), 2)])
            #print(f"Found polygon: {polygons[-1]}")

    if not polygons:
        raise ValueError("未找到任何多边形信息")

    return polygons, scale_factor

def draw_polygons(polygons, scale_factor, output_path):

    min_x = min(min(p[0] for p in poly) for poly in polygons)
    max_x = max(max(p[0] for p in poly) for poly in polygons)
    min_y = min(min(p[1] for p in poly) for poly in polygons)
    max_y = max(max(p[1] for p in poly) for poly in polygons)

    width = (max_x - min_x)
    height = (max_y - min_y)

    image = np.zeros((height, width, 3), dtype=np.uint8)

    for polygon in tqdm(polygons, desc="Drawing Polygons"):
        scaled_polygon = np.array([(int((p[0] - min_x)), int((p[1] - min_y))) for p in polygon])
        cv2.fillPoly(image, [scaled_polygon], (255, 255, 255))

    cv2.imwrite(output_path, image)

def glp_to_png(glp_file_path, png_file_path):
    try:
        polygons, scale_factor = parse_glp_file(glp_file_path)
        draw_polygons(polygons, scale_factor, png_file_path)
    except ValueError as e:
        print(f"Error: {e}")

glp_file_path = 'glp/alu_45.glp'
png_file_path = 'image/alu_45_output.png'
glp_to_png(glp_file_path, png_file_path)