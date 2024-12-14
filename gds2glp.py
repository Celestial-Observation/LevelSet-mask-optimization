import gdspy
from tqdm import tqdm

def gds_to_gl1(gds_path, gl1_path, micron_per_unit=0.001, target_layer=11):
    
    lib = gdspy.GdsLibrary()
    lib.read_gds(gds_path)
    
    cells = lib.top_level()
    if len(cells) == 0:
        print("GDS文件没有有效单元格。")
        return

    cell = cells[0]
    cell_name = cell.name

    lines = []
    lines.append(f"BEGIN     /* GDS TO GL1 GENERATED */")
    lines.append(f"EQUIV  1  {int(1/micron_per_unit)}  MICRON  +X,+Y")
    lines.append(f"CNAME {cell_name}")
    lines.append(f"LEVEL M1")
    
    polygons_by_spec = cell.get_polygons(by_spec=True)
    
    target_polygons = polygons_by_spec.get((target_layer, 0), [])
    total_polygons = len(target_polygons)
    
    with tqdm(total=total_polygons, desc="Processing Polygons") as pbar:
        for polygon in target_polygons:
            points = [(int(x / micron_per_unit), int(y / micron_per_unit)) for x, y in polygon]
            coords = " ".join(f"{x} {y}" for x, y in points)
            lines.append(f"   PGON N M1  {coords}")
            pbar.update(1)
    
    lines.append("ENDMSG")

    with open(gl1_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"GL1文件已保存到: {gl1_path}")

gds_to_gl1("gds/alu_45.gds", "glp/alu_45.glp", target_layer=11)