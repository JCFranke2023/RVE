"""
Abaqus Python script to create a voxelized cuboid model for diffusion analysis with
spherical inclusions from CSV coordinate data.
"""
import traceback
from abaqus import *
from abaqusConstants import *
import regionToolset
import mesh
import csv
import numpy as np

# ====== PARAMETERS (MODIFY AS NEEDED) ======
# Model dimensions
MODEL_LENGTH = 5  # X dimension
MODEL_WIDTH = 5   # Y dimension
MODEL_HEIGHT = 5  # Z dimension

# Voxelization parameters
VOXEL_SIZE = 0.2      # Size of each voxel element

# Material properties
MATRIX_DIFFUSIVITY = 1.0e-9  # Diffusion coefficient for matrix
MATRIX_SOLUBILITY = 0.5      # Solubility coefficient for matrix

# Boundary conditions - concentrations
TOP_CONCENTRATION = 1.0      # Normalized concentration at top surface
BOTTOM_CONCENTRATION = 0.0   # Normalized concentration at bottom surface

# Analysis parameters
TOTAL_TIME = 1000.0          # Total simulation time
INITIAL_TIME_INCREMENT = 10.0 # Initial time increment

# File path for inclusion data
CSV_FILE_PATH = 'C:/Users/franke/source/repos/JCFranke2023/RVE/inclusions.csv'  # Format: x,y,z,radius

# ====== HELPER FUNCTIONS ======
def read_inclusion_data(file_path):
    """Read inclusion data from CSV file"""
    inclusions = []
    radii = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        next(reader, None)  # Skip header if present
        for row in reader:
            x, y, z, r = float(row[0]), float(row[1]), float(row[2]), float(row[4]) # csv file contains 5 columns, data on implementation depth is not needed here
            inclusions.append((x, y, z, r))
            if r not in radii:
                radii.append(r)
    return inclusions, radii

def create_materials(model, radii):
    """Create materials for matrix and inclusions with diffusion properties"""
    # Matrix material with diffusion properties
    matrix_mat = model.Material(name='Matrix_Material')
    matrix_mat.Diffusivity(table=((MATRIX_DIFFUSIVITY, ), ))
    matrix_mat.Solubility(table=((MATRIX_SOLUBILITY, ), ))
    
    # Create matrix section
    model.HomogeneousSolidSection(name='Matrix_Section', 
                                  material='Matrix_Material', 
                                  thickness=None)    
    
    # Inclusion materials with radius-dependent properties
    for radius in radii:
        radius_str = str(radius).replace(".", "_")
        material_name = f'Inclusion_Material_{radius_str}'
        
        # Scale diffusivity and solubility based on radius
        # Adjust these scaling factors based on your specific material behavior
        diffusivity_factor = 0.1 + (radius / 10.0)  # Example scaling: larger inclusions have higher diffusivity
        solubility_factor = 1.5 - (radius / 20.0)   # Example scaling: larger inclusions have lower solubility
        
        incl_mat = model.Material(name=material_name)
        incl_mat.Diffusivity(table=((MATRIX_DIFFUSIVITY * diffusivity_factor, ), ))
        incl_mat.Solubility(table=((MATRIX_SOLUBILITY * solubility_factor, ), ))
    
        model.HomogeneousSolidSection(name=f'Inclusion_Section_{radius_str}', 
                                      material=material_name, 
                                      thickness=None)
    return

def assign_elements_to_inclusions(model, part, assembly, inclusions, radii):
    """Identify elements within each inclusion and assign appropriate material"""
    # Get all elements in the mesh
    all_elements = part.elements
    print(f"Total number of elements: {len(all_elements)}")
    
    # Track assigned elements to avoid overlapping assignments
    assigned_elements = set()
    
    # Process inclusions in order of decreasing radius (larger inclusions first)
    inclusion_data = [(idx, x, y, z, r) for idx, (x, y, z, r) in enumerate(inclusions)]
    sorted_inclusions = sorted(inclusion_data, key=lambda x: x[4], reverse=True)
    
    # Create sets for each inclusion
    for idx, x, y, z, radius in sorted_inclusions:
        # Create a set name for this inclusion
        set_name = f'Inclusion_Set_{idx}'
        
        # Find elements whose centroids are within this sphere
        inclusion_elements = []
        
        for element in all_elements:
            # Skip already assigned elements
            if element.label in assigned_elements:
                continue
                
            # Get element centroid
            centroid = element.getCentroid(GLOBAL)
            centroid_x, centroid_y, centroid_z = centroid[0][0], centroid[0][1], centroid[0][2]
            
            # Calculate distance from centroid to inclusion center
            distance = ((centroid_x - x)**2 + (centroid_y - y)**2 + (centroid_z - z)**2)**0.5
            
            # If centroid is within inclusion radius, add to set
            if distance <= radius:
                inclusion_elements.append(element)
                assigned_elements.add(element.label)  # Mark as assigned
        
        # If we found elements within this inclusion, create a set and assign material
        if inclusion_elements:
            # Create a set in the part (not the assembly)
            part.Set(elements=mesh.MeshElementArray(inclusion_elements), name=set_name)
            
            # Reference the set in the assembly instance
            assembly_region = assembly.instances['Cuboid-1'].sets[set_name]
            
            # Assign the appropriate material section to this inclusion set
            radius_str = str(radius).replace(".", "_")
            assembly.SectionAssignment(region=assembly_region,
                                      sectionName=f'Inclusion_Section_{radius_str}')
            
            print(f"Assigned {len(inclusion_elements)} elements to inclusion {idx} with radius {radius}")
    
    # Report stats
    print(f"Total assigned elements: {len(assigned_elements)} of {len(all_elements)}")
    return

# ====== MAIN MODEL CREATION ======
def create_voxelized_diffusion_model():
    """Create the voxelized cuboid model for diffusion analysis with spherical inclusions"""
    # Start a new model
    Mdb()
    model = mdb.Model(name='VoxelizedDiffusionModel')
    
    # Create the main cuboid part
    s = model.ConstrainedSketch(name='cuboid_sketch', sheetSize=200.0)
    s.rectangle(point1=(0.0, 0.0), point2=(MODEL_LENGTH, MODEL_WIDTH))
    cuboid_part = model.Part(name='Cuboid', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    cuboid_part.BaseSolidExtrude(sketch=s, depth=MODEL_HEIGHT)
    
    # Read inclusion data
    inclusions, radii = read_inclusion_data(CSV_FILE_PATH)
    print(f"Loaded {len(inclusions)} inclusions from CSV with {len(radii)} unique radii")
        
    # Create materials and sections
    create_materials(model, radii)
    
    # Create voxelized mesh
    elem_type = mesh.ElemType(elemCode=DC3D8)
    cuboid_part.setElementType(regions=(cuboid_part.cells,), elemTypes=(elem_type,))
    cuboid_part.seedPart(size=VOXEL_SIZE)
    cuboid_part.generateMesh()
    
    # Create assembly
    assembly = model.rootAssembly
    cuboid_instance = assembly.Instance(name='Cuboid-1', part=cuboid_part, dependent=ON)
    
    # Create a set for the entire model (initially all matrix material)
    all_cells = cuboid_instance.cells
    assembly.Set(cells=all_cells, name='All_Cells')
    
    # Assign matrix material to all cells initially
    region = assembly.sets['All_Cells']
    assembly.SectionAssignment(region=region, sectionName='Matrix_Section')
    
    # Assign elements to inclusions with the fixed function
    assign_elements_to_inclusions(model, cuboid_part, assembly, inclusions, radii)
    
    # Create a mass diffusion step
    model.MassDiffusionStep(name='DiffusionStep', 
                           previous='Initial',
                           timePeriod=TOTAL_TIME, 
                           maxNumInc=1000,
                           initialInc=INITIAL_TIME_INCREMENT,
                           minInc=1e-5,
                           maxInc=INITIAL_TIME_INCREMENT*10)
    
    # Define concentration boundary conditions
    # Bottom face with concentration = 0
    bottom_face = assembly.instances['Cuboid-1'].faces.getByBoundingBox(
        xMin=-0.1, yMin=-0.1, zMin=-0.1,
        xMax=MODEL_LENGTH+0.1, yMax=MODEL_WIDTH+0.1, zMax=0.1)
    assembly.Set(faces=bottom_face, name='Bottom_Face')
    model.ConcentrationBC(name='Bottom_Concentration', 
                         createStepName='DiffusionStep',
                         region=assembly.sets['Bottom_Face'],
                         magnitude=BOTTOM_CONCENTRATION,
                         distributionType=UNIFORM)
    
    # Top face with concentration = 1
    top_face = assembly.instances['Cuboid-1'].faces.getByBoundingBox(
        xMin=-0.1, yMin=-0.1, zMin=MODEL_HEIGHT-0.1,
        xMax=MODEL_LENGTH+0.1, yMax=MODEL_WIDTH+0.1, zMax=MODEL_HEIGHT+0.1)
    assembly.Set(faces=top_face, name='Top_Face')
    model.ConcentrationBC(name='Top_Concentration', 
                         createStepName='DiffusionStep',
                         region=assembly.sets['Top_Face'],
                         magnitude=TOP_CONCENTRATION,
                         distributionType=UNIFORM)
    
    # Set initial conditions (zero concentration throughout)
    all_nodes = assembly.instances['Cuboid-1'].nodes
    assembly.Set(nodes=all_nodes, name='All_Nodes')
    model.Concentration(name='Initial_Concentration',
                       createStepName='Initial',
                       region=assembly.sets['All_Nodes'],
                       magnitude=0.0,
                       distributionType=UNIFORM)
    
    # Define output requests
    model.FieldOutputRequest(name='F-Output-1',
                            createStepName='DiffusionStep',
                            variables=('CONC', 'CFLUX', 'COORD'),
                            frequency=10)
    
    # Save the model
    mdb.saveAs('voxelized_diffusion_model.cae')
    print("Diffusion model created successfully!")

# Execute the main function when script is run
if __name__ == "__main__":
    try:
        create_voxelized_diffusion_model()
    except Exception as e:
        traceback.print_exc()
