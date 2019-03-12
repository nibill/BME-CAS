# An example from scipy cookbook demonstrating the use of numpy arrys in vtk

import vtk
import numpy as np

data_matrix = np.load('segmentation.npy')
data_matrix = data_matrix.astype(np.uint8)
print(np.min(np.min(data_matrix)))
print(np.max(np.max(data_matrix)))

dataImporter = vtk.vtkImageImport()
data_string = data_matrix.tostring()
dataImporter.CopyImportVoidPointer(data_string, len(data_string))
dataImporter.SetDataScalarTypeToUnsignedChar()
dataImporter.SetNumberOfScalarComponents(1)
[h, w, z] = data_matrix.shape
dataImporter.SetDataExtent(0, z - 1, 0, w - 1, 0, h - 1)
dataImporter.SetWholeExtent(0, z - 1, 0, w - 1, 0, h - 1)
dataImporter.SetDataSpacing(1.6445319652557373, 1.6445319652557373, 1.25)

colors = vtk.vtkNamedColors()
colors.SetColor("Pelvis", [255, 255, 255, 255])
colors.SetColor("Vertebraes", [0, 255, 0, 255])
colors.SetColor("Spinal cord", [255, 255, 0, 255])
colors.SetColor("Discs", [255, 0, 0, 255])


def extract(color, isovalue):
    skinExtractor = vtk.vtkDiscreteMarchingCubes()
    skinExtractor.SetInputConnection(dataImporter.GetOutputPort())
    skinExtractor.SetValue(0, isovalue)

    smooth = vtk.vtkSmoothPolyDataFilter()
    smooth.SetInputConnection(skinExtractor.GetOutputPort())
    smooth.SetNumberOfIterations(15)
    smooth.SetRelaxationFactor(0.2)
    smooth.FeatureEdgeSmoothingOff()
    smooth.BoundarySmoothingOn()
    smooth.Update()

    skinStripper = vtk.vtkStripper()
    skinStripper.SetInputConnection(smooth.GetOutputPort())

    skinMapper = vtk.vtkOpenGLPolyDataMapper()
    skinMapper.SetInputConnection(skinStripper.GetOutputPort())
    skinMapper.ScalarVisibilityOff()

    skin = vtk.vtkOpenGLActor()
    skin.SetMapper(skinMapper)
    skin.GetProperty().SetDiffuseColor(colors.GetColor3d(color))
    skin.GetProperty().SetSpecular(.3)
    skin.GetProperty().SetSpecularPower(20)
    return skin


renderer = vtk.vtkOpenGLRenderer()
# renderer.AddActor(extract("Vertebraes", 1))
renderer.AddActor(extract("Pelvis", 3))
renderer.AddActor(extract("Spinal cord", 1))
renderer.AddActor(extract("Discs", 4))
renderer.SetBackground(0, 0, 0)

renderWin = vtk.vtkRenderWindow()
renderWin.AddRenderer(renderer)
renderInteractor = vtk.vtkRenderWindowInteractor()
style = vtk.vtkInteractorStyleMultiTouchCamera()
renderInteractor.SetInteractorStyle(style)
renderInteractor.SetRenderWindow(renderWin)

renderWin.SetSize(400, 400)

# Launching the renderer
renderInteractor.Initialize()
renderWin.Render()
renderInteractor.Start()
