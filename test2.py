"""
Modified Abaqus Python script to create a voxelized cuboid model for diffusion analysis with
spherical inclusions from CSV coordinate data and periodic boundary conditions.
"""
import traceback
from abaqus import *
import regionToolset
import mesh
import csv
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
from odbAccess import *
#from abaqusConstants import SS findet er nicht????
from abaqusConstants import *

'''
==================== Offene Fragen ===================
- Periodic BCs
- end=SS
- History Output
'''
# ==================== SIMULATION PARAMETERS ====================
# --- Model Dimensions ---
MODEL_LENGTH = 5       # X dimension of the model (mm)
MODEL_WIDTH = 5        # Y dimension of the model (mm)
MODEL_HEIGHT = 5       # Z dimension of the model (mm)

# --- Meshing Parameters ---
VOXEL_SIZE = 0.2       # Size of each voxel element (mm) - controls mesh refinement

# --- Material Properties ---
MATRIX_DIFFUSIVITY = 1.0e-4  # Diffusion coefficient for matrix (mm^2/s)
MATRIX_SOLUBILITY = 0.5      # Solubility coefficient for matrix

# --- Inclusion Scaling Factors ---
# These factors scale diffusivity and solubility based on inclusion radius
DIFF_SCALING_FACTOR = 100.0  # Multiplier for diffusivity scaling
DIFF_SCALING_RADIUS = 10.0   # Reference radius for diffusivity scaling
SOL_SCALING_OFFSET = 1.5     # Base value for solubility scaling
SOL_SCALING_RADIUS = 20.0    # Reference radius for solubility scaling

# --- Boundary Conditions ---
TOP_CONCENTRATION = 100.0    # Concentration at top surface (z_max)
BOTTOM_CONCENTRATION = 0.0   # Concentration at bottom surface (z_min)
PERIODIC_X_Y = False          # Enable periodicity in x and y directions (thin film model)

# --- Analysis Parameters ---
TOTAL_TIME = 100000.0        # Total simulation time (s)
INITIAL_TIME_INCREMENT = 10.0 # Initial time increment (s)
MIN_TIME_INCREMENT = 1e-5    # Minimum allowed time increment (s)
MAX_TIME_INCREMENT = 100.0   # Maximum allowed time increment (s) = INITIAL_TIME_INCREMENT * 10
DCMAX = 0.8                  # Maximum allowed concentration change per increment

# --- Output Parameters ---
OUTPUT_FREQUENCY = 10        # Frequency of field output requests

# --- Numerical Parameters ---
NODE_MATCH_TOLERANCE = 1e-6  # Tolerance for node matching in periodic boundary conditions

# --- File Paths ---
CSV_FILE_PATH = 'C:/Users/franke/Desktop/Neuer Ordner/inclusions.csv'  # Format: x,y,z,radius
OUTPUT_CAE_PATH = 'voxelized_diffusion_model_with_PBC.cae'  # Output CAE file name

# --- Debugging Options ---
INCLUSION_LIMIT = 10         # Limit number of inclusions to process (for faster debugging)
WORKING_DIRECTORY = 'C:/Users/franke/Desktop/Neuer Ordner/'  # Working directory path
# ============================================================

# Change to the directory where the script is located
os.chdir(WORKING_DIRECTORY)

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
            if count >= INCLUSION_LIMIT:  # Limit for faster debugging
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
        
        # Scale diffusivity and solubility based on radius using the specified scaling factors
        diffusivity_factor = DIFF_SCALING_FACTOR * (radius / DIFF_SCALING_RADIUS)
        solubility_factor = SOL_SCALING_OFFSET - (radius / SOL_SCALING_RADIUS)
        
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

def create_face_node_sets(model_name, assembly, instance_name):
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
    
    # In Abaqus, zero flux is the default for all boundaries, so we don't need 
    # to explicitly set flux boundary conditions.
    
    print(f"Created face sets for {instance_name}")
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
        return 0
    
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
            # For mass diffusion, the DOF is 11 (concentration DOF in Abaqus)
            model.Equation(name=f'Periodic_Constraint_{constraint_count}', 
                          terms=((1.0, target_node.instanceName, target_node.label, 11), 
                                (-1.0, source_node.instanceName, source_node.label, 11)))
            constraint_count += 1
    
    print(f"Created {constraint_count} periodic constraints between {source_set_name} and {target_set_name}")
    return constraint_count

def create_film_periodic_constraints(model, assembly, instance_name):
    """Create comprehensive periodic constraints for thin film (x and y periodic only)"""
    if not PERIODIC_X_Y:
        print("Periodic boundary conditions in x and y directions are disabled.")
        return 0
        
    # Identify and group all nodes on the boundary
    x_min, x_max = 0.0, MODEL_LENGTH
    y_min, y_max = 0.0, MODEL_WIDTH
    tol = NODE_MATCH_TOLERANCE
    
    instance = assembly.instances[instance_name]
    
    # Function to check if a node is on a specific boundary
    def is_on_boundary(node, boundary):
        x, y, z = node.coordinates
        if boundary == 'x_min': return abs(x - x_min) < tol
        if boundary == 'x_max': return abs(x - x_max) < tol
        if boundary == 'y_min': return abs(y - y_min) < tol
        if boundary == 'y_max': return abs(y - y_max) < tol
        return False
    
    # Identify corners, edges, and faces
    corners = {}
    x_edges = {}  # Dictionary: (y_type, z) -> [nodes] (edges along x direction)
    y_edges = {}  # Dictionary: (x_type, z) -> [nodes] (edges along y direction)
    
    print("Categorizing boundary nodes...")
    
    # Categorize all nodes on the boundaries
    for node in instance.nodes:
        x, y, z = node.coordinates
        # Round z to handle floating point comparison
        z_rounded = round(z, 6)
        
        # Check if node is on any boundary
        on_x_min = is_on_boundary(node, 'x_min')
        on_x_max = is_on_boundary(node, 'x_max')
        on_y_min = is_on_boundary(node, 'y_min')
        on_y_max = is_on_boundary(node, 'y_max')
        
        # If node is not on any boundary, skip it
        if not (on_x_min or on_x_max or on_y_min or on_y_max):
            continue
            
        # Check if node is at a corner
        if (on_x_min or on_x_max) and (on_y_min or on_y_max):
            x_type = 'x_max' if on_x_max else 'x_min'
            y_type = 'y_max' if on_y_max else 'y_min'
            corner_key = (x_type, y_type, z_rounded)
            corners[corner_key] = node
        # Check if node is on an x-direction edge (constant y)
        elif not (on_x_min or on_x_max) and (on_y_min or on_y_max):
            y_type = 'y_max' if on_y_max else 'y_min'
            edge_key = (y_type, z_rounded)
            if edge_key not in x_edges:
                x_edges[edge_key] = []
            x_edges[edge_key].append(node)
        # Check if node is on a y-direction edge (constant x)
        elif (on_x_min or on_x_max) and not (on_y_min or on_y_max):
            x_type = 'x_max' if on_x_max else 'x_min'
            edge_key = (x_type, z_rounded)
            if edge_key not in y_edges:
                y_edges[edge_key] = []
            y_edges[edge_key].append(node)
    
    # Debug info
    print(f"Found {len(corners)} corner nodes")
    print(f"Found {len(x_edges)} x-direction edges")
    print(f"Found {len(y_edges)} y-direction edges")
    
    # Create constraint counter
    constraint_count = 0
    
    # 1. Handle corners - map to their partners
    print("Creating corner constraints...")
    for corner_key, corner_node in corners.items():
        x_type, y_type, z = corner_key
        
        # Skip (x_min, y_min, z) corners (our reference corners)
        if x_type == 'x_min' and y_type == 'y_min':
            continue
        
        # Find the corresponding reference corner
        ref_key = ('x_min', 'y_min', z)
        if ref_key in corners:
            ref_node = corners[ref_key]
            
            # Create constraint
            model.Equation(name=f'Corner_Constraint_{constraint_count}', 
                          terms=((1.0, corner_node.instanceName, corner_node.label, 11), 
                                (-1.0, ref_node.instanceName, ref_node.label, 11)))
            constraint_count += 1
    
    print(f"Created {constraint_count} corner constraints")
    
    # 2. Handle x-direction edges (along constant y)
    print("Creating x-edge constraints...")
    edge_constraints = 0
    for (y_type, z), edge_nodes in x_edges.items():
        # Skip y_min edges (our reference edges)
        if y_type == 'y_min':
            continue
            
        # Find corresponding nodes on y_min edge
        ref_key = ('y_min', z)
        if ref_key in x_edges:
            ref_nodes = x_edges[ref_key]
            
            # Sort nodes by x-coordinate
            edge_nodes.sort(key=lambda n: n.coordinates[0])
            ref_nodes.sort(key=lambda n: n.coordinates[0])
            
            # Create constraints for matching nodes
            for i in range(min(len(edge_nodes), len(ref_nodes))):
                model.Equation(name=f'X_Edge_Constraint_{constraint_count}', 
                              terms=((1.0, edge_nodes[i].instanceName, edge_nodes[i].label, 11), 
                                    (-1.0, ref_nodes[i].instanceName, ref_nodes[i].label, 11)))
                constraint_count += 1
                edge_constraints += 1
    
    print(f"Created {edge_constraints} x-edge constraints")
    
    # 3. Handle y-direction edges (along constant x)
    print("Creating y-edge constraints...")
    edge_constraints = 0
    for (x_type, z), edge_nodes in y_edges.items():
        # Skip x_min edges (our reference edges)
        if x_type == 'x_min':
            continue
            
        # Find corresponding nodes on x_min edge
        ref_key = ('x_min', z)
        if ref_key in y_edges:
            ref_nodes = y_edges[ref_key]
            
            # Sort nodes by y-coordinate
            edge_nodes.sort(key=lambda n: n.coordinates[1])
            ref_nodes.sort(key=lambda n: n.coordinates[1])
            
            # Create constraints for matching nodes
            for i in range(min(len(edge_nodes), len(ref_nodes))):
                model.Equation(name=f'Y_Edge_Constraint_{constraint_count}', 
                              terms=((1.0, edge_nodes[i].instanceName, edge_nodes[i].label, 11), 
                                    (-1.0, ref_nodes[i].instanceName, ref_nodes[i].label, 11)))
                constraint_count += 1
                edge_constraints += 1
    
    print(f"Created {edge_constraints} y-edge constraints")
    
    # 4. Create node sets for the faces (these will exclude edges and corners)
    print("Creating face node sets...")
    create_face_node_sets(model.name, assembly, instance_name)
    
    # 5. Filter face nodes to exclude edges and corners
    print("Filtering face nodes and creating face constraints...")
    
    # Identify nodes on edges and corners
    edge_corner_nodes = set()
    for nodes_list in x_edges.values():
        for node in nodes_list:
            edge_corner_nodes.add((node.instanceName, node.label))
    for nodes_list in y_edges.values():
        for node in nodes_list:
            edge_corner_nodes.add((node.instanceName, node.label))
    for node in corners.values():
        edge_corner_nodes.add((node.instanceName, node.label))
    
    # Function to filter out edge and corner nodes
    def filter_edge_corner_nodes(node_set):
        filtered_nodes = []
        for node in node_set:
            if (node.instanceName, node.label) not in edge_corner_nodes:
                filtered_nodes.append(node)
        return filtered_nodes
    
    # Apply x-direction face constraints
    face_constraints = 0
    try:
        x_min_nodes = filter_edge_corner_nodes(assembly.sets['x_min_nodes'].nodes)
        x_max_nodes = filter_edge_corner_nodes(assembly.sets['x_max_nodes'].nodes)
        
        # Create a dictionary of x_min node positions
        x_min_dict = {}
        for node in x_min_nodes:
            key = (round(node.coordinates[1], 6), round(node.coordinates[2], 6))
            x_min_dict[key] = node
        
        # Create constraints between matching face nodes
        for node in x_max_nodes:
            key = (round(node.coordinates[1], 6), round(node.coordinates[2], 6))
            if key in x_min_dict:
                model.Equation(name=f'X_Face_Constraint_{constraint_count}', 
                              terms=((1.0, node.instanceName, node.label, 11), 
                                    (-1.0, x_min_dict[key].instanceName, x_min_dict[key].label, 11)))
                constraint_count += 1
                face_constraints += 1
    except KeyError:
        print("Warning: Could not find x face node sets")
    
    # Apply y-direction face constraints
    try:
        y_min_nodes = filter_edge_corner_nodes(assembly.sets['y_min_nodes'].nodes)
        y_max_nodes = filter_edge_corner_nodes(assembly.sets['y_max_nodes'].nodes)
        
        # Create a dictionary of y_min node positions
        y_min_dict = {}
        for node in y_min_nodes:
            key = (round(node.coordinates[0], 6), round(node.coordinates[2], 6))
            y_min_dict[key] = node
        
        # Create constraints between matching face nodes
        for node in y_max_nodes:
            key = (round(node.coordinates[0], 6), round(node.coordinates[2], 6))
            if key in y_min_dict:
                model.Equation(name=f'Y_Face_Constraint_{constraint_count}', 
                              terms=((1.0, node.instanceName, node.label, 11), 
                                    (-1.0, y_min_dict[key].instanceName, y_min_dict[key].label, 11)))
                constraint_count += 1
                face_constraints += 1
    except KeyError:
        print("Warning: Could not find y face node sets")
    
    print(f"Created {face_constraints} face constraints")
    print(f"Created a total of {constraint_count} periodic constraints for the thin film model")
    return constraint_count

# ======= FLUX VISUALIZATION  =======

def extract_flux_data(odb_path):
    """
    Extract flux data at the top face from the Abaqus ODB file
    
    Args:
        odb_path (str): Path to the output database file
    
    Returns:
        tuple: (time_points, avg_flux_values) arrays for plotting
    """
    try:
        # Open the output database
        print(f"Opening ODB file: {odb_path}")
        odb = openOdb(path=odb_path, readOnly=True)
        
        # Get step data
        step = odb.steps['DiffusionStep']
        
        # Get the top face set from the assembly
        top_face_set = odb.rootAssembly.nodeSets['TOP_FACE_NODES']
        
        # Collect time and flux data
        time_points = []
        avg_flux_values = []
        
        # Loop through all frames
        print("Extracting flux data from frames...")
        for frame_idx, frame in enumerate(step.frames):
            # Get the frame time
            time = frame.frameValue
            time_points.append(time)
            
            # Get NJFLUX (nodal mass flux) field output
            if 'NJFLUX' in frame.fieldOutputs:
                flux_field = frame.fieldOutputs['NJFLUX']
                
                # Get flux values for the top face
                top_flux = flux_field.getSubset(region=top_face_set)
                
                # We're interested in the z-component of flux (index 2) at the top face
                z_flux_values = [value.data[2] for value in top_flux.values]
                
                # Calculate average flux
                avg_flux = sum(z_flux_values) / len(z_flux_values)
                avg_flux_values.append(avg_flux)
            else:
                print(f"Warning: NJFLUX not found in frame {frame_idx}")
                avg_flux_values.append(0.0)
        
        # Close the ODB
        odb.close()
        
        return np.array(time_points), np.array(avg_flux_values)
    except Exception as e:
        print(f"Error in extract_flux_data: {str(e)}")
        traceback.print_exc()
        return np.array([]), np.array([])

def create_flux_plot(time_points, flux_values, output_path):
    """
    Create and save a plot of flux vs time
    
    Args:
        time_points (np.array): Array of time points
        flux_values (np.array): Array of flux values
        output_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, flux_values, 'b-', linewidth=2)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Average Flux (mass/area/time)', fontsize=12)
    plt.title('Mass Flux through Top Surface vs Time', fontsize=14)
    plt.grid(True)
    
    # Add annotations for interesting points
    max_idx = np.argmax(np.abs(flux_values))
    plt.annotate(f'Max Flux: {flux_values[max_idx]:.3e}',
                xy=(time_points[max_idx], flux_values[max_idx]),
                xytext=(time_points[max_idx] * 0.8, flux_values[max_idx] * 1.2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # Add steady-state annotation if reached
    if len(time_points) > 10:
        steady_state_value = flux_values[-1]
        plt.axhline(y=steady_state_value, color='r', linestyle='--', alpha=0.7)
        plt.annotate(f'Steady State: {steady_state_value:.3e}',
                    xy=(time_points[-1], steady_state_value),
                    xytext=(time_points[-1] * 0.7, steady_state_value * 1.1),
                    arrowprops=dict(facecolor='red', shrink=0.05, width=1.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Flux plot saved to: {output_path}")
    
    # Also save data to a CSV file for further analysis
    csv_path = output_path.replace('.png', '.csv')
    np.savetxt(csv_path, np.column_stack((time_points, flux_values)), 
              delimiter=',', header='Time(s),Flux(mass/area/time)', comments='')
    print(f"Flux data saved to: {csv_path}")
    
    return plt.gcf()  # Return the figure for potential display

# ====== ADDITIONAL CODE TO CREATE NODE SETS ======
# ===== ADD HISTORY OUTPUT REQUEST =====
def add_flux_history_output(model):
    """Add a history output request for flux data"""
    model.HistoryOutputRequest(name='H-Output-Flux',
                              createStepName='DiffusionStep',
                              #variables=('MFLUX',),  # Use MFLUX for mass flux
                              region=model.rootAssembly.surfaces['TOP_FACE_SURFACE'],
                              sectionPoints=DEFAULT,
                              rebar=EXCLUDE)

def create_additional_node_sets(model, assembly, instance_name):
    instance = assembly.instances[instance_name]
    
    # Keep the node set for any node-based outputs
    top_nodes = instance.nodes.getByBoundingBox(
        xMin=float(-NODE_MATCH_TOLERANCE), xMax=float(MODEL_LENGTH+NODE_MATCH_TOLERANCE),
        yMin=float(-NODE_MATCH_TOLERANCE), yMax=float(MODEL_WIDTH+NODE_MATCH_TOLERANCE),
        zMin=float(MODEL_HEIGHT-NODE_MATCH_TOLERANCE), zMax=float(MODEL_HEIGHT+NODE_MATCH_TOLERANCE))
    assembly.Set(nodes=top_nodes, name='TOP_FACE_NODES')
    
    # Add a surface for flux measurements
    top_faces = instance.faces.getByBoundingBox(
        xMin=float(-NODE_MATCH_TOLERANCE), xMax=float(MODEL_LENGTH+NODE_MATCH_TOLERANCE),
        yMin=float(-NODE_MATCH_TOLERANCE), yMax=float(MODEL_WIDTH+NODE_MATCH_TOLERANCE),
        zMin=float(MODEL_HEIGHT-NODE_MATCH_TOLERANCE), zMax=float(MODEL_HEIGHT+NODE_MATCH_TOLERANCE))
    assembly.Surface(side1Faces=top_faces, name='TOP_FACE_SURFACE')
    print(f"Created TOP_FACE_SURFACE with {len(top_faces)} faces for flux measurement")

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
                            initialInc=float(INITIAL_TIME_INCREMENT),
                            timePeriod=float(TOTAL_TIME),
                            dcmax=float(DCMAX),  # Maximum concentration change per increment
                            end=1.0)
                           #maxNumInc=100000,
                           #minInc=float(MIN_TIME_INCREMENT),
                           #maxInc=float(MAX_TIME_INCREMENT))#,
    
    # Apply face sets for boundary conditions
    apply_periodic_boundary_conditions(model, assembly, instance_name)
    
    # Create periodic boundary conditions with special treatment for edges and corners
    if PERIODIC_X_Y:
        print("\n=== Creating Periodic Boundary Conditions ===")
        total_constraints = create_film_periodic_constraints(model, assembly, instance_name)
        print(f"Total periodic constraints created: {total_constraints}")
    else:
        print("Skipping periodic boundary conditions as PERIODIC_X_Y is disabled.")
    
    # Define concentration boundary conditions
    # Bottom face with concentration = 0
    bottom_face = assembly.instances[instance_name].faces.getByBoundingBox(
        xMin=float(-0.1), yMin=float(-0.1), zMin=float(-0.1),
        xMax=float(MODEL_LENGTH+0.1), yMax=float(MODEL_WIDTH+0.1), zMax=float(0.1))
    assembly.Set(faces=bottom_face, name='Bottom_Face')
    model.ConcentrationBC(name='Bottom_Concentration', 
                         createStepName='DiffusionStep',
                         region=assembly.sets['Bottom_Face'],
                         magnitude=float(BOTTOM_CONCENTRATION),
                         distributionType=UNIFORM)
    
    # Top face with concentration = TOP_CONCENTRATION
    top_face = assembly.instances[instance_name].faces.getByBoundingBox(
        xMin=float(-0.1), yMin=float(-0.1), zMin=float(MODEL_HEIGHT-0.1),
        xMax=float(MODEL_LENGTH+0.1), yMax=float(MODEL_WIDTH+0.1), zMax=float(MODEL_HEIGHT+0.1))
    assembly.Set(faces=top_face, name='Top_Face')
    model.ConcentrationBC(name='Top_Concentration', 
                         createStepName='DiffusionStep',
                         region=assembly.sets['Top_Face'],
                         magnitude=float(TOP_CONCENTRATION),
                         distributionType=UNIFORM)
    
    # Set initial conditions (zero concentration throughout)
    all_nodes = assembly.instances[instance_name].nodes
    assembly.Set(nodes=all_nodes, name='All_Nodes')
    
    create_additional_node_sets(model, assembly, instance_name)
    add_flux_history_output(model)

    # Create initial concentration field
    model.Temperature(name='Initial_Concentration',
                     createStepName='Initial', 
                     region=assembly.sets['All_Nodes'],
                     distributionType=UNIFORM, 
                     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS,
                     magnitudes=(float(0.0),))
    
    # Define output requests with correct variables for mass diffusion analysis
    model.FieldOutputRequest(name='F-Output-1',
                            createStepName='DiffusionStep',
                            variables=('CONC', 'FLUC', 'NJFLUX'),  # Added FLUC and NJFLUX
                            frequency=OUTPUT_FREQUENCY)
    
    # Add additional node sets for flux extraction
    create_additional_node_sets(model, assembly, instance_name)
    
    # Add history output request for flux
    add_flux_history_output(model)
    
    # Save the model
    mdb.saveAs(OUTPUT_CAE_PATH)
    print("Diffusion model with periodic boundary conditions created successfully!")
    
    # Run the job and create a plot if successful
    try:
        # Create and submit the job
        job_name = 'VoxelizedDiffusionModel_Job'
        mdb.Job(name=job_name, model=model.name, description='Diffusion analysis with flux extraction')
        print(f"Submitting job: {job_name}")
        mdb.jobs[job_name].submit()
        mdb.jobs[job_name].waitForCompletion()
        
        # Check if job completed successfully
        if mdb.jobs[job_name].status == COMPLETED:
            print("Job completed successfully, extracting flux data...")
            # Extract flux data and create plot
            odb_path = f"{job_name}.odb"
            time_points, flux_values = extract_flux_data(odb_path)
            
            if len(time_points) > 0:
                create_flux_plot(time_points, flux_values, f"{WORKING_DIRECTORY}/flux_vs_time.png")
                print("Flux analysis completed and plot generated!")
            else:
                print("No flux data was extracted, check logs for errors.")
        else:
            print(f"Job did not complete successfully. Status: {mdb.jobs[job_name].status}")
    except Exception as e:
        print(f"Error running job or creating flux plot: {str(e)}")
        traceback.print_exc()

# Execute the main function when script is run
if __name__ == "__main__":
    try:
        create_voxelized_diffusion_model()
    except Exception as e:
        traceback.print_exc()
