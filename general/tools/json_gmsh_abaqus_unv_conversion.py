import json
import os
import re

def gmsh_formating(filename):
    with open(filename, 'r') as fr:
        result = json.loads(fr.read())

    points = []
    lines = []
    for id, point in enumerate(result):
        points.append(f"Point({id+1}) = {{{point[0]/100}, {point[1]/100}, {0}, {1.0}}};")
        if id == len(result) - 1:
            lines.append(f"Line({id+1}) = {{{id+1}, {1}}};")
        else:
            lines.append(f"Line({id+1}) = {{{id+1}, {id+2}}};")

    with open(filename+'.txt', 'w') as fw:
        for point in points:
            fw.write(point + "\n")
        for line in lines:
            fw.write(line + "\n")
    print()

# gmsh_formating("domains/test3.json")

def boundary_domains(file_name):
    with open(file_name, 'r') as fr:
        result = json.loads(fr.read())

    print(result)


def read_inp_file(inp_file):
    """ Reads the .inp file and extracts node and element data """
    nodes = []
    elements = []

    with open(inp_file, 'r') as f:
        lines = f.readlines()

        reading_nodes = False
        reading_elements = False

        for line in lines:
            # Detect where the node data starts
            if '*Node' in line:
                reading_nodes = True
                reading_elements = False
                continue

            # Detect where the element data starts
            if '*Element' in line:
                reading_elements = True
                reading_nodes = False
                continue

            # Read node data
            if reading_nodes:
                if line.startswith('*'):
                    reading_nodes = False
                    continue
                parts = line.strip().split(',')
                node_id = int(parts[0])
                coordinates = [float(c) for c in parts[1:]]
                nodes.append((node_id, coordinates))

            # Read element data
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
    """ Writes nodes and elements to a .unv file """
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
    """ Main function to convert .inp file to .unv format """
    nodes, elements = read_inp_file(inp_file)
    write_unv_file(unv_file, nodes, elements)

# Example usage:
inp_to_unv('D:\\Onedrive\\OneDrive - University of Calgary\\Research & work\\AI\\AI EDAM 2024\\domains\shapes\\tkde\\g_d1.inp', 'model.unv')

# boundary_domains("domains/boundary_fly_r2.json")