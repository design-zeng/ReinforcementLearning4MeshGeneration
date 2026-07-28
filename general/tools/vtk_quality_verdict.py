import math
import re
from pathlib import Path

import vtk
import meshio

# meshes are produced by sac/infer.py evaluation() (save_fig=True)
root = Path(__file__).parent.parent.parent / "sac" / "output" / "evaluation"

# Auto-discover the produced meshes (relative stems under `root`).
domains = sorted(str(p.relative_to(root))[:-4] for p in root.rglob("*.inp"))

def render(domain):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(f"{root}/{domain}.vtk")
    reader.Update()

    colors = vtk.vtkNamedColors()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create a rendering window and renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    ren.SetBackground(colors.GetColor3d("cobalt_green"))

    # Create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Assign actor to the renderer
    ren.AddActor(actor)

    # Enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()

def DumpQualityStats(iq, arrayname):
    an = iq.GetOutput().GetFieldData().GetArray(arrayname)
    cardinality = an.GetComponent(0, 4)
    range = list()
    range.append(an.GetComponent(0, 0))
    range.append(an.GetComponent(0, 2))
    average = an.GetComponent(0, 1)
    stdDev = math.sqrt(math.fabs(an.GetComponent(0, 3)))
    outStr = '%s%g%s%g%s%g\n%s%g%s%g' % (
        '  cardinality: ', cardinality,
        '  , range: ', range[0], '  -  ', range[1],
        '  average: ', average, '  , standard deviation: ', stdDev)
    return outStr

def verdict(domains):
    metrics = {
        'QualityMeasureToMinAngle': [],
        'QualityMeasureToMaxAngle': [],
        'QualityMeasureToScaledJacobian': [],
        'QualityMeasureToStretch': [],
        'QualityMeasureToTaper': [],
    }
    for d in domains:
        verdict_domain(d, metrics)
    for k, v in metrics.items():
        print(k, sum([_v[0] for _v in v]) / len(v), sum([_v[1] for _v in v]) / len(v))

def verdict_domain(domain, metrics):
    filename = f"{root}/{domain}.inp"
    mr = vtk.vtkUnstructuredGridReader()

    iq = vtk.vtkMeshQuality()

    m = meshio.Mesh.read(filename, "abaqus")
    m.write(f"{root}/{domain}.vtk")  # \\gmsh

    mr.SetFileName(f"{root}/{domain}.vtk")
    mr.Update()

    ug = mr.GetOutput()
    iq.SetInputConnection(mr.GetOutputPort())

    # Here we define the various mesh types and labels for output.
    meshTypes = [
        ['Quad', 'Quadrilateral',
            [
                ['QualityMeasureToMinAngle', ' Minimal Angle:'],
                ['QualityMeasureToMaxAngle', ' Maximum Angle:'],
                ['QualityMeasureToScaledJacobian', ' Scaled Jacobian:'],
                ['QualityMeasureToStretch', ' Stretch:'],
                ['QualityMeasureToTaper', ' Taper:'],
            ]
        ],
    ]

    if ug.GetNumberOfCells() > 0:
        res = ''
        for meshType in meshTypes:
            if meshType[0] == 'Tet':
                res += '\n%s%s\n   %s' % ('Tetrahedral',
                                          ' quality of the mesh:', mr.GetFileName())
            elif meshType[0] == 'Hex':
                res += '\n%s%s\n   %s' % ('Hexahedral',
                                          ' quality of the mesh:', mr.GetFileName())
            else:
                res += '\n%s%s\n   %s' % (meshType[1],
                                          ' quality of the mesh:', mr.GetFileName())

            for measure in meshType[2]:
                eval('iq.Set' + meshType[0] + measure[0] + '()')
                iq.Update()
                res += '\n%s\n%s' % (measure[1],
                                     DumpQualityStats(iq, 'Mesh ' + meshType[1] + ' Quality'))
                p = re.compile('average: [0-9.]+')
                str = p.search(DumpQualityStats(iq, 'Mesh ' + meshType[1] + ' Quality')).group()
                ave = str.split(' ')[1]
                p = re.compile('standard deviation: [0-9.]+')
                str = p.search(DumpQualityStats(iq, 'Mesh ' + meshType[1] + ' Quality')).group()
                std = str.split(' ')[2]
                metrics[measure[0]].append([float(ave), float(std)])

            res += '\n'
        return metrics

if __name__ == '__main__':
    if not domains:
        raise FileNotFoundError(
            f"No .inp meshes under {root}. Run `python -m sac.infer` (save_fig=True) first.")
    verdict(domains)
