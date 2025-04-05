"""
Modified Abaqus Python script to create a voxelized cuboid model for diffusion analysis with
spherical inclusions from CSV coordinate data and periodic boundary conditions.
"""
import traceback
from abaqus import *
from abaqusConstants import *
import regionToolset
import mesh
import csv
import numpy as np

import os

# ====== PARAMETERS (MODIFY AS NEEDED) ======
# Model dimensions
MODEL_LENGTH = 5  # X dimension
MODEL_WIDTH = 5   # Y dimension
MODEL_HEIGHT = 5  # Z dimension

# Voxelization parameters
VOXEL_SIZE = 0.2      # Size of each voxel element

# Material properties
MATRIX_DIFFUSIVITY = 1.0e-7  # Diffusion coefficient for matrix
MATRIX_SOLUBILITY = 0.5      # Solubility coefficient for matrix

# Boundary conditions - concentrations
TOP_CONCENTRATION = 100.0      # Normalized concentration at top surface
BOTTOM_CONCENTRATION = 0.0   # Normalized concentration at bottom surface

# Analysis parameters
TOTAL_TIME = 100000.0          # Total simulation time
INITIAL_TIME_INCREMENT = 10.0 # Initial time increment

# Tolerance for node matching in periodic boundary conditions
NODE_MATCH_TOLERANCE = 1e-6

# File path for inclusion data
CSV_FILE_PATH = 'C:/Users/franke/Desktop/Neuer Ordner/inclusions.csv'  # Format: x,y,z,radius

# Change to the directory where the script is located
os.chdir('C:/Users/franke/Desktop/Neuer Ordner/')

# ====== HELPER FUNCTIONS ======
def read_inclusion_data(file_path):
    """Read inclusion data from CSV file"""
    inclusions = []
    radii = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        next(reader, None)  # Skip header if present
        count = 0
        for row in reader:
            x, y, z, r = float(row[0]), float(row[1]), float(row[2]), float(row[4]) # csv file contains 5 columns, data on implementation depth is not needed here
            inclusions.append((x, y, z, r))
            if r not in radii:
                radii.append(r)
            count += 1
            if count >= 10:  # Limit to 10 inclusions for faster debugging
                break
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
        diffusivity_factor = 100 * (radius / 10.0)  # Example scaling: larger inclusions have higher diffusivity
        solubility_factor = 1.5 - (radius / 20.0)   # Example scaling: larger inclusions have lower solubility
        
        incl_mat = model.Material(name=material_name)
        incl_mat.Diffusivity(table=((MATRIX_DIFFUSIVITY * diffusivity_factor, ), ))
        incl_mat.Solubility(table=((MATRIX_SOLUBILITY * solubility_factor, ), ))
    
        model.HomogeneousSolidSection(name=f'Inclusion_Section_{radius_str}', 
                                      material=material_name, 
                                      thickness=None)
    return

def assign_materials_to_part(part, assembly, inclusions, radii):
    """Assign materials to part before meshing"""
    # Create a set for the entire part (matrix material)
    all_cells = part.cells
    part.Set(cells=all_cells, name='Matrix_Set')
    part.SectionAssignment(region=part.sets['Matrix_Set'], sectionName='Matrix_Section')
    
    # After meshing, we'll handle the inclusions as element sets
    return

def assign_elements_to_inclusions(part, assembly_instance, inclusions, radii):
    """Identify elements within each inclusion and assign appropriate material"""
    # Get all elements in the mesh
    all_elements = part.elements
    print(f"Total number of elements: {len(all_elements)}")
    
    # Get element centroids (using element connectivity and node coordinates)
    element_centroids = {}
    for element in all_elements:
        # Get nodes of the element
        element_nodes = element.getNodes()
        # Calculate centroid as average of node coordinates
        sum_x, sum_y, sum_z = 0.0, 0.0, 0.0
        for node in element_nodes:
            coords = node.coordinates
            sum_x += coords[0]
            sum_y += coords[1]
            sum_z += coords[2]
        num_nodes = len(element_nodes)
        element_centroids[element.label] = (sum_x/num_nodes, sum_y/num_nodes, sum_z/num_nodes)
    
    # Track assigned elements to avoid overlapping assignments
    assigned_elements = set()
    
    # Process inclusions in order of decreasing radius (larger inclusions first)
    inclusion_data = [(idx, x, y, z, r) for idx, (x, y, z, r) in enumerate(inclusions)]
    sorted_inclusions = sorted(inclusion_data, key=lambda x: x[4], reverse=True)
    
    # Create sets for each inclusion and assign materials
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
            centroid_x, centroid_y, centroid_z = element_centroids[element.label]
            
            # Calculate distance from centroid to inclusion center
            distance = ((centroid_x - x)**2 + (centroid_y - y)**2 + (centroid_z - z)**2)**0.5
            
            # If centroid is within inclusion radius, add to set
            if distance <= radius:
                inclusion_elements.append(element)
                assigned_elements.add(element.label)  # Mark as assigned
        
        # If we found elements within this inclusion, create a set and assign material
        if inclusion_elements:
            # Create a set in the part
            element_array = mesh.MeshElementArray(inclusion_elements)
            part.Set(elements=element_array, name=set_name)
            
            # Set the section for this inclusion set
            radius_str = str(radius).replace(".", "_")
            section_name = f'Inclusion_Section_{radius_str}'
            
            # Assign the section to the element set at the part level
            part.SectionAssignment(region=part.sets[set_name], sectionName=section_name)
            
            print(f"Assigned {len(inclusion_elements)} elements to inclusion {idx} with radius {radius}")
    
    # Report stats
    print(f"Total assigned elements: {len(assigned_elements)} of {len(all_elements)}")
    return

def create_face_node_sets(part, assembly, instance_name):
    """Create node sets for each face of the cuboid for periodic boundary conditions"""
    try:
        instance = assembly.instances[instance_name]
        print(f"Successfully found instance: {instance_name}")
        
        # Get node count for debugging
        all_nodes = instance.nodes
        print(f"Total nodes in instance: {len(all_nodes)}")
        
        # Define the six faces of the cuboid
        faces = {
            'x_min': instance.nodes.getByBoundingBox(
                xMin=-NODE_MATCH_TOLERANCE, xMax=NODE_MATCH_TOLERANCE,
                yMin=-NODE_MATCH_TOLERANCE, yMax=MODEL_WIDTH+NODE_MATCH_TOLERANCE,
                zMin=-NODE_MATCH_TOLERANCE, zMax=MODEL_HEIGHT+NODE_MATCH_TOLERANCE),
            'x_max': instance.nodes.getByBoundingBox(
                xMin=MODEL_LENGTH-NODE_MATCH_TOLERANCE, xMax=MODEL_LENGTH+NODE_MATCH_TOLERANCE,
                yMin=-NODE_MATCH_TOLERANCE, yMax=MODEL_WIDTH+NODE_MATCH_TOLERANCE, 
                zMin=-NODE_MATCH_TOLERANCE, zMax=MODEL_HEIGHT+NODE_MATCH_TOLERANCE),
            'y_min': instance.nodes.getByBoundingBox(
                xMin=-NODE_MATCH_TOLERANCE, xMax=MODEL_LENGTH+NODE_MATCH_TOLERANCE,
                yMin=-NODE_MATCH_TOLERANCE, yMax=NODE_MATCH_TOLERANCE,
                zMin=-NODE_MATCH_TOLERANCE, zMax=MODEL_HEIGHT+NODE_MATCH_TOLERANCE),
            'y_max': instance.nodes.getByBoundingBox(
                xMin=-NODE_MATCH_TOLERANCE, xMax=MODEL_LENGTH+NODE_MATCH_TOLERANCE,
                yMin=MODEL_WIDTH-NODE_MATCH_TOLERANCE, yMax=MODEL_WIDTH+NODE_MATCH_TOLERANCE,
                zMin=-NODE_MATCH_TOLERANCE, zMax=MODEL_HEIGHT+NODE_MATCH_TOLERANCE),
            'z_min': instance.nodes.getByBoundingBox(
                xMin=-NODE_MATCH_TOLERANCE, xMax=MODEL_LENGTH+NODE_MATCH_TOLERANCE,
                yMin=-NODE_MATCH_TOLERANCE, yMax=MODEL_WIDTH+NODE_MATCH_TOLERANCE,
                zMin=-NODE_MATCH_TOLERANCE, zMax=NODE_MATCH_TOLERANCE),
            'z_max': instance.nodes.getByBoundingBox(
                xMin=-NODE_MATCH_TOLERANCE, xMax=MODEL_LENGTH+NODE_MATCH_TOLERANCE,
                yMin=-NODE_MATCH_TOLERANCE, yMax=MODEL_WIDTH+NODE_MATCH_TOLERANCE,
                zMin=MODEL_HEIGHT-NODE_MATCH_TOLERANCE, zMax=MODEL_HEIGHT+NODE_MATCH_TOLERANCE)
        }
        
        # Create node sets for each face
        for face_name, face_nodes in faces.items():
            # Check if we found any nodes
            if len(face_nodes) == 0:
                print(f"WARNING: No nodes found for face '{face_name}'")
                continue
            
            # Create the set
            try:
                assembly.Set(nodes=face_nodes, name=f'{face_name}_nodes')
                print(f"Created node set '{face_name}_nodes' with {len(face_nodes)} nodes")
            except Exception as e:
                print(f"Error creating set for '{face_name}': {str(e)}")
        
        return faces
    except Exception as e:
        print(f"Error in create_face_node_sets: {str(e)}")
        traceback.print_exc()
        return {}

def apply_periodic_boundary_conditions(model, assembly, instance_name):
    """Apply periodic boundary conditions to the side faces using a simpler approach"""
    instance = assembly.instances[instance_name]
    
    # Create sets for side faces using face objects rather than nodes
    # Left face (x=0)
    left_faces = instance.faces.getByBoundingBox(
        xMin=-0.1, xMax=0.1,
        yMin=-0.1, yMax=MODEL_WIDTH+0.1,
        zMin=-0.1, zMax=MODEL_HEIGHT+0.1)
    assembly.Set(faces=left_faces, name='Left_Face_Set')
    print(f"Created left face set with {len(left_faces)} faces")
    
    # Right face (x=MODEL_LENGTH)
    right_faces = instance.faces.getByBoundingBox(
        xMin=MODEL_LENGTH-0.1, xMax=MODEL_LENGTH+0.1,
        yMin=-0.1, yMax=MODEL_WIDTH+0.1,
        zMin=-0.1, zMax=MODEL_HEIGHT+0.1)
    assembly.Set(faces=right_faces, name='Right_Face_Set')
    print(f"Created right face set with {len(right_faces)} faces")
    
    # Front face (y=0)
    front_faces = instance.faces.getByBoundingBox(
        xMin=-0.1, xMax=MODEL_LENGTH+0.1,
        yMin=-0.1, yMax=0.1,
        zMin=-0.1, zMax=MODEL_HEIGHT+0.1)
    assembly.Set(faces=front_faces, name='Front_Face_Set')
    print(f"Created front face set with {len(front_faces)} faces")
    
    # Back face (y=MODEL_WIDTH)
    back_faces = instance.faces.getByBoundingBox(
        xMin=-0.1, xMax=MODEL_LENGTH+0.1,
        yMin=MODEL_WIDTH-0.1, yMax=MODEL_WIDTH+0.1,
        zMin=-0.1, zMax=MODEL_HEIGHT+0.1)
    assembly.Set(faces=back_faces, name='Back_Face_Set')
    print(f"Created back face set with {len(back_faces)} faces")
    
    # Create mass diffusion flux conditions (zero flux) for side faces
    # In Abaqus, zero flux is the default for all boundaries, so we don't need 
    # to explicitly set flux boundary conditions. By not specifying any boundary
    # conditions on the side faces, they will automatically have zero flux.
    
    print(f"Applied zero-flux boundary conditions to the side faces of {instance_name}")
    print("Note: In Abaqus, faces without explicit boundary conditions have zero flux by default.")
    return

def create_periodic_constraints(model, assembly, source_set_name, target_set_name, coordinate_index):
    """Create equation constraints between corresponding nodes on opposite faces"""
    # Check if sets exist and get nodes
    try:
        source_nodes = assembly.sets[source_set_name].nodes
        target_nodes = assembly.sets[target_set_name].nodes
    except KeyError:
        print(f"Error: Node sets '{source_set_name}' or '{target_set_name}' do not exist in the assembly")
        print(f"Available sets: {assembly.sets.keys()}")
        return
    
    # Get the instance name from the first node
    instance_name = source_nodes[0].instanceName
    
    # Create a dictionary of source node positions for efficient lookup
    source_dict = {}
    for node in source_nodes:
        # Create a key based on the coordinates perpendicular to the pairing direction
        if coordinate_index == 0:  # x direction
            key = (round(node.coordinates[1], 6), round(node.coordinates[2], 6))
        elif coordinate_index == 1:  # y direction
            key = (round(node.coordinates[0], 6), round(node.coordinates[2], 6))
        else:  # z direction (not used for periodic BCs in this case)
            key = (round(node.coordinates[0], 6), round(node.coordinates[1], 6))
        
        source_dict[key] = node
    
    # Match target nodes with source nodes and create constraints
    constraint_count = 0
    for target_node in target_nodes:
        # Create the key for this target node
        if coordinate_index == 0:  # x direction
            key = (round(target_node.coordinates[1], 6), round(target_node.coordinates[2], 6))
        elif coordinate_index == 1:  # y direction
            key = (round(target_node.coordinates[0], 6), round(target_node.coordinates[2], 6))
        else:  # z direction
            key = (round(target_node.coordinates[0], 6), round(target_node.coordinates[1], 6))
        
        # Find the corresponding source node
        if key in source_dict:
            source_node = source_dict[key]
            
            # Create an equation constraint for the concentration degree of freedom
            # For mass diffusion, the DOF is 1 (concentration DOF in Abaqus)
            model.Equation(name=f'Periodic_Constraint_{constraint_count}', 
                          terms=((1.0, target_node.instanceName, target_node.label, 1), 
                                (-1.0, source_node.instanceName, source_node.label, 1)))
            constraint_count += 1
    
    print(f"Created {constraint_count} periodic constraints between {source_set_name} and {target_set_name}")
    return

# ====== MAIN MODEL CREATION ======
def create_voxelized_diffusion_model():
    """Create the voxelized cuboid model for diffusion analysis with spherical inclusions"""
    # Start a new model
    #Mdb()
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
    
    # Assign matrix material to the entire part before meshing
    assign_materials_to_part(cuboid_part, model.rootAssembly, inclusions, radii)
    
    # Create voxelized mesh
    elem_type = mesh.ElemType(elemCode=DC3D8)
    cuboid_part.setElementType(regions=(cuboid_part.cells,), elemTypes=(elem_type,))
    cuboid_part.seedPart(size=VOXEL_SIZE)
    cuboid_part.generateMesh()
    
    # Create assembly
    assembly = model.rootAssembly
    instance_name = 'Cuboid-1'
    cuboid_instance = assembly.Instance(name=instance_name, part=cuboid_part, dependent=ON)
    
    # Assign elements to inclusions
    assign_elements_to_inclusions(cuboid_part, cuboid_instance, inclusions, radii)
    
    # Create a mass diffusion step
    model.MassDiffusionStep(name='DiffusionStep', 
                           previous='Initial',
                           timePeriod=TOTAL_TIME, 
                           maxNumInc=100000,
                           initialInc=INITIAL_TIME_INCREMENT,
                           minInc=1e-5,
                           maxInc=INITIAL_TIME_INCREMENT*10,
                           dcmax=0.1)  # Add maximum concentration change per increment
    
    # Apply periodic boundary conditions to side faces
    apply_periodic_boundary_conditions(model, assembly, instance_name)
    
    # Define concentration boundary conditions
    # Bottom face with concentration = 0
    bottom_face = assembly.instances[instance_name].faces.getByBoundingBox(
        xMin=-0.1, yMin=-0.1, zMin=-0.1,
        xMax=MODEL_LENGTH+0.1, yMax=MODEL_WIDTH+0.1, zMax=0.1)
    assembly.Set(faces=bottom_face, name='Bottom_Face')
    model.ConcentrationBC(name='Bottom_Concentration', 
                         createStepName='DiffusionStep',
                         region=assembly.sets['Bottom_Face'],
                         magnitude=BOTTOM_CONCENTRATION,
                         distributionType=UNIFORM)
    
    # Top face with concentration = 1
    top_face = assembly.instances[instance_name].faces.getByBoundingBox(
        xMin=-0.1, yMin=-0.1, zMin=MODEL_HEIGHT-0.1,
        xMax=MODEL_LENGTH+0.1, yMax=MODEL_WIDTH+0.1, zMax=MODEL_HEIGHT+0.1)
    assembly.Set(faces=top_face, name='Top_Face')
    model.ConcentrationBC(name='Top_Concentration', 
                         createStepName='DiffusionStep',
                         region=assembly.sets['Top_Face'],
                         magnitude=TOP_CONCENTRATION,
                         distributionType=UNIFORM)
    
    # Set initial conditions (zero concentration throughout)
    all_nodes = assembly.instances[instance_name].nodes
    assembly.Set(nodes=all_nodes, name='All_Nodes')
    
    # Create initial concentration field
    model.Temperature(name='Initial_Concentration',
                     createStepName='Initial', 
                     region=assembly.sets['All_Nodes'],
                     distributionType=UNIFORM, 
                     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS,
                     magnitudes=(0.0,))
    
    # Define output requests with correct variables for mass diffusion analysis
    model.FieldOutputRequest(name='F-Output-1',
                            createStepName='DiffusionStep',
                            variables=('CONC', 'NT'),
                            frequency=10)
    
    # Save the model
    mdb.saveAs('voxelized_diffusion_model_with_PBC.cae')
    print("Diffusion model with periodic boundary conditions created successfully!")

# Execute the main function when script is run
if __name__ == "__main__":
    try:
        create_voxelized_diffusion_model()
    except Exception as e:
        traceback.print_exc()