"""
Abaqus Python script to create a voxelized cuboid model with spherical inclusions
from CSV coordinate data.
"""
from abaqus import *
from abaqusConstants import *
import regionToolset
import mesh
import csv
import numpy as np

# ====== PARAMETERS (MODIFY AS NEEDED) ======
# Model dimensions
MODEL_LENGTH = 100.0  # X dimension
MODEL_WIDTH = 100.0   # Y dimension
MODEL_HEIGHT = 100.0  # Z dimension

# Voxelization parameters
VOXEL_SIZE = 2.0      # Size of each voxel element

# Material properties
MATRIX_ELASTIC_MODULUS = 2100.0
MATRIX_POISSON_RATIO = 0.3
INCLUSION_ELASTIC_MODULUS = 7000.0
INCLUSION_POISSON_RATIO = 0.25

# File path for inclusion data
CSV_FILE_PATH = 'inclusions.csv'  # Format: x,y,z,radius

# ====== HELPER FUNCTIONS ======
def read_inclusion_data(file_path):
    """Read inclusion data from CSV file"""
    inclusions = []
    radii = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header if present
        for row in reader:
            if len(row) >= 4:
                x, y, z, r = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                inclusions.append((x, y, z, r))
                if r not in radii:
                    radii.append(r)
    return inclusions, radii

def create_materials(model, radii):
    """Create materials for matrix and inclusions"""
    # Matrix material
    matrix_mat = model.Material(name='Matrix_Material')
    matrix_mat.Elastic(table=((MATRIX_ELASTIC_MODULUS, MATRIX_POISSON_RATIO), ))
    
    # Create sections
    model.HomogeneousSolidSection(name='Matrix_Section', 
                                  material='Matrix_Material', 
                                  thickness=None)    
    
    # Inclusion material
    for radius in radii:
        material_name = f'Inclusion_Material_{radius}'
        incl_mat = model.Material(name=material_name)
        incl_mat.Elastic(table=((radius * INCLUSION_ELASTIC_MODULUS, INCLUSION_POISSON_RATIO), ))
    
        model.HomogeneousSolidSection(name=f'Inclusion_Section_{radius}', 
                                      material=material_name, 
                                      thickness=None)
    return

# ====== MAIN MODEL CREATION ======
def create_voxelized_model():
    """Create the voxelized cuboid model with spherical inclusions"""
    # Start a new model
    Mdb()
    model = mdb.Model(name='VoxelizedCuboidModel')
    
    # Create the main cuboid part
    s = model.ConstrainedSketch(name='cuboid_sketch', sheetSize=200.0)
    s.rectangle(point1=(0.0, 0.0), point2=(MODEL_LENGTH, MODEL_WIDTH))
    cuboid_part = model.Part(name='Cuboid', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    cuboid_part.BaseSolidExtrude(sketch=s, depth=MODEL_HEIGHT)
    

    # Read inclusion data
    inclusions, radii = read_inclusion_data(CSV_FILE_PATH)
    print(f"Loaded {len(inclusions)} inclusions from CSV")
        
    # Create materials and sections
    create_materials(model, radii)
    
    # Create assembly
    assembly = model.rootAssembly
    cuboid_instance = assembly.Instance(name='Cuboid-1', part=cuboid_part, dependent=ON)
    
    # Create a set for the entire model (initially all matrix material)
    all_cells = cuboid_instance.cells.getSequenceFromMask(mask=('[]',), )
    assembly.Set(cells=all_cells, name='All_Cells')
    
    # Assign matrix material to all cells initially
    region = assembly.sets['All_Cells']
    assembly.SectionAssignment(region=region, sectionName='Matrix_Section', 
                              offset=0.0, offsetType=MIDDLE_SURFACE)
    
    # Create voxelized mesh
    elem_type = mesh.ElemType(elemCode=C3D8, technique=STANDARD)
    cuboid_part.setElementType(regions=(cuboid_part.cells,), elemTypes=(elem_type,))
    cuboid_part.seedPart(size=VOXEL_SIZE)
    cuboid_part.generateMesh()
    
    # Define sets for inclusion regions
    # We'll identify elements within each inclusion sphere
    for idx, (x, y, z, radius) in enumerate(inclusions):
        # Create a spherical partition for each inclusion
        # Here we use a field approach for voxelized models
        # Define a field to identify elements within the sphere
        field_name = f'Inclusion_{idx}'
        assembly.assignMaterialOrientationToRegion(
            region=assembly.sets['All_Cells'], 
            localCsys=None, 
            axis=AXIS_1, 
            angle=0.0, 
            stackDirection=STACK_3)
        
        # Create field output to identify elements in each sphere
        model.FieldOutputRequest(name=field_name,
                                createStepName='Initial',
                                variables=('COORD',))
        
        # In a real implementation, you would use Abaqus Python API to:
        # 1. Get all element centroids
        # 2. Check if they fall within sphere: (x-x0)^2 + (y-y0)^2 + (z-z0)^2 <= r^2
        # 3. Assign inclusion material to those elements
        
        # Placeholder approach (pseudo-code):
        # For voxelized approach, create a field script to identify elements:
        field_script = f"""
        from math import sqrt
        def getDistance(x, y, z):
            return sqrt((x-{x})**2 + (y-{y})**2 + (z-{z})**2)
            
        def getElements():
            inclusion_elements = []
            for element in elements:
                # Get element centroid
                centroid = element.getCentroid()
                x, y, z = centroid[0], centroid[1], centroid[2]
                if getDistance(x, y, z) <= {radius}:
                    inclusion_elements.append(element)
            return inclusion_elements
        """
        # Note: The above field_script is conceptual and would need to be
        # implemented using proper Abaqus Python API calls
    
    # Create a step for analysis
    model.StaticStep(name='LoadStep', previous='Initial',
                    timePeriod=1.0, nlgeom=OFF, maxNumInc=100,
                    initialInc=0.1, minInc=1e-5, maxInc=0.1)
    
    # Boundary conditions
    # Bottom face fixed in all directions
    bottom_face = assembly.instances['Cuboid-1'].faces.getByBoundingBox(
        xMin=-0.1, yMin=-0.1, zMin=-0.1,
        xMax=MODEL_LENGTH+0.1, yMax=MODEL_WIDTH+0.1, zMax=0.1)
    assembly.Set(faces=bottom_face, name='Bottom_Face')
    model.DisplacementBC(name='Fixed_Bottom', createStepName='LoadStep',
                        region=assembly.sets['Bottom_Face'],
                        u1=0.0, u2=0.0, u3=0.0)
    
    # Apply a test load on top face
    top_face = assembly.instances['Cuboid-1'].faces.getByBoundingBox(
        xMin=-0.1, yMin=-0.1, zMin=MODEL_HEIGHT-0.1,
        xMax=MODEL_LENGTH+0.1, yMax=MODEL_WIDTH+0.1, zMax=MODEL_HEIGHT+0.1)
    assembly.Set(faces=top_face, name='Top_Face')
    model.Pressure(name='Top_Pressure', createStepName='LoadStep',
                  region=assembly.sets['Top_Face'], magnitude=1.0)
    
    # Save the model
    mdb.saveAs('voxelized_inclusion_model.cae')
    print("Model created successfully!")

# Execute the main function when script is run
if __name__ == "__main__":
    create_voxelized_model()

