import json


def json_to_gmsh(filename):
    with open(filename, 'r') as fr:
        result = json.loads(fr.read())

    points = []
    lines = []
    for idx, point in enumerate(result):
        points.append(f"Point({idx+1}) = {{{point[0]/100}, {point[1]/100}, {0}, {1.0}}};")
        if idx == len(result) - 1:
            lines.append(f"Line({idx+1}) = {{{idx+1}, {1}}};")
        else:
            lines.append(f"Line({idx+1}) = {{{idx+1}, {idx+2}}};")

    with open(filename+'.txt', 'w') as fw:
        for point in points:
            fw.write(point + "\n")
        for line in lines:
            fw.write(line + "\n")


def read_inp_file(inp_file):
    nodes = []
    elements = []

    with open(inp_file, 'r') as f:
        lines = f.readlines()

        reading_nodes = False
        reading_elements = False

        for line in lines:
            if '*Node' in line:
                reading_nodes = True
                reading_elements = False
                continue

            if '*Element' in line:
                reading_elements = True
                reading_nodes = False
                continue

            if reading_nodes:
                if line.startswith('*'):
                    reading_nodes = False
                    continue
                parts = line.strip().split(',')
                node_id = int(parts[0])
                coordinates = [float(c) for c in parts[1:]]
                nodes.append((node_id, coordinates))

            if reading_elements:
                if line.startswith('*'):
                    reading_elements = False
                    continue
                parts = line.strip().split(',')
                element_id = int(parts[0])
                node_ids = [int(n) for n in parts[1:]]
                elements.append((element_id, node_ids))

    return nodes, elements


def write_unv_file(unv_file, nodes, elements):
    with open(unv_file, 'w') as f:
        # Write header for nodes (Universal file format section 2411)
        f.write("    -1\n")
        f.write("  2411\n")
        for node in nodes:
            node_id = node[0]
            x, y, z = node[1]
            f.write(f"{node_id:10d}{x:20.10e}{y:20.10e}{z:20.10e}\n")
        f.write("    -1\n")

        # Write header for elements (Universal file format section 2412)
        f.write("    -1\n")
        f.write("  2412\n")
        for element in elements:
            element_id = element[0]
            node_ids = element[1]
            # Assume solid element type 11 (generic for this example)
            element_type = 11
            f.write(f"{element_id:10d}{element_type:10d}{'':10s}")
            f.write("".join([f"{n:10d}" for n in node_ids]))
            f.write("\n")
        f.write("    -1\n")


def inp_to_unv(inp_file, unv_file):
    nodes, elements = read_inp_file(inp_file)
    write_unv_file(unv_file, nodes, elements)


if __name__ == '__main__':
    import os
    from pathlib import Path
    base_path = Path(__file__).parent.parent.parent
    output_path = base_path / "general" / "output"
    os.makedirs(output_path, exist_ok=True)

    # A domain (JSON list of [x, y] points) -> Gmsh geometry. json_to_gmsh writes
    # <input>.txt next to its input, so copy the domain into output/ first to keep
    # samples/ read-only.
    import shutil
    domain_copy = output_path / "basic1.json"
    shutil.copy(base_path / "samples" / "domains" / "basic1.json", domain_copy)
    json_to_gmsh(str(domain_copy))
    print("Wrote", output_path / "basic1.json.txt")

    # An Abaqus .inp mesh -> UNV. .inp meshes are produced by sac/infer.py, so run
    # `python -m sac.infer` (save_fig=True) first; here we convert the first one.
    meshes = sorted((base_path / "sac" / "output" / "evaluation").rglob("*.inp"))
    if meshes:
        inp_to_unv(str(meshes[0]), str(output_path / (meshes[0].stem + ".unv")))
        print("Wrote", output_path / (meshes[0].stem + ".unv"))
    else:
        print("No produced .inp meshes yet; run `python -m sac.infer` (save_fig=True).")
