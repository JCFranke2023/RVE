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
from abaqusConstants import *

# ==================== SIMULATION PARAMETERS ====================
# --- Model Dimensions ---
CUBE_SIZE = 5       # dimension of the model (nm)

# --- Meshing Parameters ---
VOXEL_SIZE = 0.2       # Size of each voxel element (nm) - controls mesh refinement

# --- Material Properties ---
MATRIX_DIFFUSIVITY = 1.0e-4  # Diffusion coefficient for matrix (nm^2/s)
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
PERIODIC_X_Y = True          # Enable periodicity in x and y directions (thin film model)

# --- Analysis Parameters ---
TOTAL_TIME = 100000.0        # Total simulation time (s)
INITIAL_TIME_INCREMENT = 10.0 # Initial time increment (s)
MIN_TIME_INCREMENT = 1e-5    # Minimum allowed time increment (s)
MAX_TIME_INCREMENT = 100.0   # Maximum allowed time increment (s) = INITIAL_TIME_INCREMENT * 10
DCMAX = 0.8                  # Maximum allowed concentration change per increment
STEADY_STATE_THRESHOLD = 0.001 # Threshold for steady-state detection (fraction of initial concentration)

# --- Output Parameters ---
OUTPUT_FREQUENCY = 10        # Frequency of field output requests

# --- Numerical Parameters ---
NODE_MATCH_TOLERANCE = 1e-6  # Tolerance for node matching in periodic boundary conditions

# --- File Paths ---
CSV_FILE_PATH = 'C:/Users/franke/source/repos/JCFranke2023/RVE/inclusions.csv'  # Format: x,y,z,radius
OUTPUT_CAE_PATH = 'voxelized_diffusion_model_with_PBC.cae'  # Output CAE file name

# --- Debugging Options ---
INCLUSION_LIMIT = 10         # Limit number of inclusions to process (for faster debugging)
WORKING_DIRECTORY = 'C:/Users/franke/source/repos/JCFranke2023/RVE/abaqus'  # Working directory path
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

#unused
def create_face_sets(assembly, instance_name):
    """Apply periodic boundary conditions to the side faces using a simpler approach"""
    instance = assembly.instances[instance_name]
    
    # Create sets for side faces using face objects rather than nodes
    # Left face (x=0)
    left_faces = instance.faces.getByBoundingBox(
        xMin=-0.1, xMax=0.1,
        yMin=-0.1, yMax=CUBE_SIZE+0.1,
        zMin=-0.1, zMax=CUBE_SIZE+0.1)
    assembly.Set(faces=left_faces, name='Left_Face_Set')
    print(f"Created left face set with {len(left_faces)} faces")
    
    # Right face (x=MODEL_LENGTH)
    right_faces = instance.faces.getByBoundingBox(
        xMin=CUBE_SIZE-0.1, xMax=CUBE_SIZE+0.1,
        yMin=-0.1, yMax=CUBE_SIZE+0.1,
        zMin=-0.1, zMax=CUBE_SIZE+0.1)
    assembly.Set(faces=right_faces, name='Right_Face_Set')
    print(f"Created right face set with {len(right_faces)} faces")
    
    # Front face (y=0)
    front_faces = instance.faces.getByBoundingBox(
        xMin=-0.1, xMax=CUBE_SIZE+0.1,
        yMin=-0.1, yMax=0.1,
        zMin=-0.1, zMax=CUBE_SIZE+0.1)
    assembly.Set(faces=front_faces, name='Front_Face_Set')
    print(f"Created front face set with {len(front_faces)} faces")
    
    # Back face (y=MODEL_WIDTH)
    back_faces = instance.faces.getByBoundingBox(
        xMin=-0.1, xMax=CUBE_SIZE+0.1,
        yMin=CUBE_SIZE-0.1, yMax=CUBE_SIZE+0.1,
        zMin=-0.1, zMax=CUBE_SIZE+0.1)
    assembly.Set(faces=back_faces, name='Back_Face_Set')
    print(f"Created back face set with {len(back_faces)} faces")
    
    print(f"Created face sets for {instance_name}")
    return
'''
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
'''
'''
# Claude version
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
            print(f"Creating constraint for corner {corner_key} with reference {ref_key}:")
            print(f"  Corner node: {corner_node.instanceName}, {corner_node.label}")
            model.Equation(name=f'Corner_Constraint_{constraint_count}', 
                          terms=((1.0, corner_node.instanceName, 11), 
                                (-1.0, ref_node.instanceName, 11)))
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
                              terms=((1.0, edge_nodes[i].instanceName, 11), 
                                    (-1.0, ref_nodes[i].instanceName, 11)))
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
                              terms=((1.0, edge_nodes[i].instanceName, 11), 
                                    (-1.0, ref_nodes[i].instanceName, 11)))
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
'''
"""
# Jonas version
def create_film_periodic_constraints(model, assembly, instance_name):
    #Create comprehensive periodic constraints for thin film (x and y periodic only)

    tol = NODE_MATCH_TOLERANCE
    instance = assembly.instances[instance_name]
    
    # Identify corners, edges, and faces
    corners = {}
    z_edges = {}  # Dictionary: (x, y) -> [nodes] (edges along z direction)
    
    print("Categorizing boundary nodes...")
    
    
    corner_coords = [(0, 0, 0),
                     (0, 0, CUBE_SIZE),
                     (0, CUBE_SIZE, 0),
                     (0, CUBE_SIZE, CUBE_SIZE),
                     (CUBE_SIZE, 0, 0),
                     (CUBE_SIZE, 0, CUBE_SIZE),
                     (CUBE_SIZE, CUBE_SIZE, 0),
                     (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)]
    '''
    half_side = CUBE_SIZE / 2
    center = (half_side, half_side, half_side)
    offsets = np.array(list(np.ndindex((2, 2, 2)))) * 2 - 1
    offsets = offsets * half_side
    '''
    # Add the center coordinates to get the absolute corner positions
    for ix, corner in enumerate(corner_coords):
        # get the node by bounding sphere of each corner
        node = instance.nodes.getByBoundingSphere(corner, tol)
        corners[corner] = node
        #print(f'Found corner node at {corner} (index {ix}): {node}')
        # get the nodes by bounding cylinder of each edge in z-direction
        if ix % 2 != 0:
            previous_corner = corner_coords[ix-1]
            nodes = instance.nodes.getByBoundingCylinder(corner, previous_corner, tol)
            z_edges[corner] = []
            for node in nodes:
                if node not in corners.values():
                    z_edges[corner].append(node)

    
    # Debug info
    print(f"Found {len(corners)} corner nodes")
    print(f"Found {len(z_edges)} z-direction edges")
    
    # Create constraint counter
    constraint_count = 0
    
    # 1. Handle corners - map to their partners
    print("Creating corner constraints...")
    # Create sets for the reference corners
    ref_key_z_min = (0, 0, 0)
    ref_node_z_min = corners[ref_key_z_min]
    corner_set_reference_z_min = f'corner_node_reference_z_min'
    assembly.Set(name=corner_set_reference_z_min, nodes=ref_node_z_min)

    ref_key_z_max = (0, 0, CUBE_SIZE)
    ref_node_z_max = corners[ref_key_z_max]
    corner_set_reference_z_max = f'corner_node_reference_z_max'
    assembly.Set(name=corner_set_reference_z_max, nodes=ref_node_z_max)

    for corner_key, corner_node in corners.items():
        x, y, z = corner_key
        
        # Skip (x_min, y_min, z) corners (our reference corners)
        if x == 0 and y == 0:
            continue
        if z == 0:
            ref_set_name = corner_set_reference_z_min
        else:
            ref_set_name = corner_set_reference_z_max
        # Create sets for the nodes
        corner_set_name = f'corner_node_{constraint_count}_1'
            
        # Create set for corner node
        assembly.Set(name=corner_set_name, nodes=corner_node)
            
        # Create set for ref node
            
        # Create constraint with properly referenced sets
        model.Equation(name=f'Corner_Constraint_{constraint_count}', 
                        terms=((1.0, corner_set_name, 11), 
                            (-1.0, ref_set_name, 11)))
        constraint_count += 1
    
    print(f"Created {constraint_count} corner constraints")
    
    # 2. Handle z-direction edges
    print("Creating z-edge constraints...")
    edge_constraints = 0
    # Create a set for the reference edge
    ref_key = (0, 0, CUBE_SIZE)
    ref_nodes = z_edges[ref_key]
    ref_set_name = f'z_edge_set_reference'
    assembly.Set(name=ref_set_name, nodes=ref_nodes)    
    for (x, y, z), edge_nodes in z_edges.items():
        # Skip (0, 0, z) edge (reference edge)
        if x == 0 and y == 0:
            continue
        '''    
        # Create constraints for matching nodes
        for i in range(len(ref_nodes)):
            # Create sets for the nodes
            edge_set_name = f'z_edge_node_{constraint_count}_1'
            ref_set_name = f'z_edge_node_{constraint_count}_2'
                
            # Create set for edge node
            assembly.Set(name=edge_set_name, nodes=mesh.MeshNodeArray([edge_nodes[i]]))
                
            # Create set for ref node
            assembly.Set(name=ref_set_name, nodes=mesh.MeshNodeArray([ref_nodes[i]]))
                
            # Create constraint
            model.Equation(name=f'Z_Edge_Constraint_{constraint_count}', 
                            terms=((1.0, edge_set_name, 11), 
                                (-1.0, ref_set_name, 11)))
            constraint_count += 1
            edge_constraints += 1
        '''
        # Create sets for the nodes
        edge_set_name = f'z_edge_set_{constraint_count}'
        assembly.Set(name=edge_set_name, nodes=edge_nodes)

        # Create constraint
        model.Equation(name=f'Z_Edge_Constraint_{constraint_count}', 
                        terms=((1.0, edge_set_name, 11), 
                            (-1.0, ref_set_name, 11)))
        constraint_count += 1
        edge_constraints += 1
    
    print(f"Created {edge_constraints} z_edge constraints")
    
    faces = create_face_node_sets(model, assembly, instance_name, corners, z_edges)
    print(f"Found {len(faces)} faces with {len(faces.values()[0])} nodes in face {faces.keys()[0]}")
    
    print(f"Created a total of {constraint_count} periodic constraints for the thin film model")
    return constraint_count
"""
def create_film_periodic_constraints(model, assembly, instance_name):
    # Create comprehensive periodic constraints for thin film (x and y periodic only)

    tol = NODE_MATCH_TOLERANCE
    instance = assembly.instances[instance_name]
    
    # Identify corners, edges, and faces
    corners = {}  # (corner_coordinates): node (corner nodes)
    z_edges = {} # (ref_corner_coordinates): [nodes] (edges along z direction)
    
    print("Categorizing boundary nodes...")
    
    corner_coords = [(0, 0, 0),
                     (0, 0, CUBE_SIZE),
                     (0, CUBE_SIZE, 0),
                     (0, CUBE_SIZE, CUBE_SIZE),
                     (CUBE_SIZE, 0, 0),
                     (CUBE_SIZE, 0, CUBE_SIZE),
                     (CUBE_SIZE, CUBE_SIZE, 0),
                     (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)]

    for ix, corner in enumerate(corner_coords):
        # get the node by bounding sphere of each corner
        left_node = instance.nodes.getByBoundingSphere(corner, tol)
        corners[corner] = left_node
        #print(f'Found corner node at {corner} (index {ix}): {left_node}, {type(left_node)=}')
        #print(f'{left_node[0]=}, {type(left_node[0])=}, {left_node[0].coordinates}')
        # get the nodes by bounding cylinder of each edge in z-direction
        if ix % 2 != 0:
            cylinder_top = (corner[0], corner[1], CUBE_SIZE-VOXEL_SIZE+NODE_MATCH_TOLERANCE)
            cylinder_bottom = (corner[0], corner[1], VOXEL_SIZE-NODE_MATCH_TOLERANCE)
            nodes = instance.nodes.getByBoundingCylinder(cylinder_top, cylinder_bottom, tol)
            z_edges[corner] = nodes
            '''for node in nodes:
                if node not in corners.values():
                    z_edges[corner].append(node)'''

    # Debug info
    print(f"Found {len(corners)} corner nodes")
    print(f"Found {len(z_edges)} z-direction edges")
    
    # Create constraint counter
    edge_constraint_count = 0
    edge_index = 0
    # 1. Handle z-direction edges
    print("Creating z-edge constraints...")
    # Create a set for the reference edge
    ref_key = (0, 0, CUBE_SIZE)
    ref_edge_nodes = z_edges[ref_key]
    for left_node in ref_edge_nodes:
        #print(f'Reference edge node: {left_node.label} at {left_node.coordinates}')
        assembly.Set(name=f'z_edge_node_reference_{left_node.label}', nodes=mesh.MeshNodeArray([left_node]))
    for (x, y, z), edge_nodes in z_edges.items():
        # Skip (0, 0, z) edge (reference edge)
        if x == 0 and y == 0:
            continue
        for ix, edge_node in enumerate(edge_nodes):
            ref_node_set = f'z_edge_node_reference_{ref_edge_nodes[ix].label}'
            edge_node_set = f'z_edge_node_{edge_node.label}'
            assembly.Set(name=edge_node_set, nodes=mesh.MeshNodeArray([edge_node]))
            # Create constraint
            model.Equation(name=f'Z_Edge_Constraint_{edge_index}_{edge_constraint_count}',
                           terms=((1.0, edge_node_set, 11),
                                  (-1.0, ref_node_set, 11)))
            edge_constraint_count += 1
        edge_index += 1
    print(f"Created {edge_constraint_count} z_edge constraints")
    print("Creating face constraints...")
    face_constraint_count = 0
    # front: y=0, back: y=CUBE_SIZE, left: x=0, right: x=CUBE_SIZE
    faces = get_face_nodes(instance)
    '''for ix, node in enumerate(faces['left']):
        print(f"Left face node: {node.label} at {node.coordinates}")
        right_node = faces['right'][ix]
        print(f"Right face node: {right_node.label} at {right_node.coordinates}")
    for ix, node in enumerate(faces['front']):
        print(f"Front face node: {node.label} at {node.coordinates}")
        right_node = faces['back'][ix]
        print(f"Back face node: {right_node.label} at {right_node.coordinates}")'''
    for ix, left_node in enumerate(faces['left']):
        left_node_set = f'left_face_node_{left_node.label}'
        assembly.Set(name=left_node_set, nodes=mesh.MeshNodeArray([left_node]))
        right_node = faces['right'][ix]
        right_node_set = f'right_face_node_{right_node.label}'
        assembly.Set(name=right_node_set, nodes=mesh.MeshNodeArray([right_node]))
        model.Equation(name=f'Face_Constraint_L{left_node.label}_R{right_node.label}',
                       terms=((1.0, left_node_set, 11),
                              (-1.0, right_node_set, 11)))
        face_constraint_count += 1
    for ix, front_node in enumerate(faces['front']):
        front_node_set = f'front_face_node_{front_node.label}'
        assembly.Set(name=front_node_set, nodes=mesh.MeshNodeArray([front_node]))
        back_node = faces['back'][ix]
        back_node_set = f'back_face_node_{back_node.label}'
        assembly.Set(name=back_node_set, nodes=mesh.MeshNodeArray([back_node]))
        model.Equation(name=f'Face_Constraint_F{front_node.label}_B{back_node.label}',
                       terms=((1.0, front_node_set, 11),
                              (-1.0, back_node_set, 11)))
        face_constraint_count += 1
    print(f"Created {face_constraint_count} face constraints")
    return edge_constraint_count + face_constraint_count

def get_face_nodes(instance):
    """Create node sets for each face of the cuboid for periodic boundary conditions"""
        # Define the six faces of the cuboid
    faces = {
        'left': instance.nodes.getByBoundingBox(
            xMin=-NODE_MATCH_TOLERANCE, 
            xMax=NODE_MATCH_TOLERANCE,
            yMin=VOXEL_SIZE-NODE_MATCH_TOLERANCE, 
            yMax=CUBE_SIZE-VOXEL_SIZE+NODE_MATCH_TOLERANCE,
            zMin=VOXEL_SIZE-NODE_MATCH_TOLERANCE, 
            zMax=CUBE_SIZE-VOXEL_SIZE+NODE_MATCH_TOLERANCE),
        'right': instance.nodes.getByBoundingBox(
            xMin=CUBE_SIZE-NODE_MATCH_TOLERANCE, 
            xMax=CUBE_SIZE+NODE_MATCH_TOLERANCE,
            yMin=VOXEL_SIZE-NODE_MATCH_TOLERANCE, 
            yMax=CUBE_SIZE-VOXEL_SIZE+NODE_MATCH_TOLERANCE,
            zMin=VOXEL_SIZE-NODE_MATCH_TOLERANCE, 
            zMax=CUBE_SIZE-VOXEL_SIZE+NODE_MATCH_TOLERANCE),
        'front': instance.nodes.getByBoundingBox(
            xMin=VOXEL_SIZE-NODE_MATCH_TOLERANCE, 
            xMax=CUBE_SIZE-VOXEL_SIZE+NODE_MATCH_TOLERANCE,
            yMin=-NODE_MATCH_TOLERANCE, 
            yMax=NODE_MATCH_TOLERANCE,
            zMin=VOXEL_SIZE-NODE_MATCH_TOLERANCE, 
            zMax=CUBE_SIZE-VOXEL_SIZE+NODE_MATCH_TOLERANCE),
        'back': instance.nodes.getByBoundingBox(
            xMin=VOXEL_SIZE-NODE_MATCH_TOLERANCE, 
            xMax=CUBE_SIZE-VOXEL_SIZE+NODE_MATCH_TOLERANCE,
            yMin=CUBE_SIZE-NODE_MATCH_TOLERANCE, 
            yMax=CUBE_SIZE+NODE_MATCH_TOLERANCE,
            zMin=VOXEL_SIZE-NODE_MATCH_TOLERANCE, 
            zMax=CUBE_SIZE-VOXEL_SIZE+NODE_MATCH_TOLERANCE),
        }
    return faces
        

# ===== ADD FIELD OUTPUT REQUEST =====
def create_output_requests(model):
    """Add field output request for flux data (proper way for mass diffusion)"""
    # For mass diffusion analysis, we should request FLUC and NJFLUX variables
    model.FieldOutputRequest(name='F-Output-1',
                             createStepName='DiffusionStep',
                             variables=('CONC', 'NNC', 'MFL', 'MFLT'),
                             frequency=OUTPUT_FREQUENCY)
    model.HistoryOutputRequest(name='H-Output-1',
                               createStepName='DiffusionStep',
                               variables=('CONC', 'NNC', 'MFL', 'MFLT'),
                               region=model.rootAssembly.sets['BOTTOM_FACE_NODES'])

def create_additional_node_sets(model, assembly, instance_name):
    """Create additional node sets for flux measurement"""
    instance = assembly.instances[instance_name]
    
    # Create a node set for the top face for flux measurements
    bottom_nodes = instance.nodes.getByBoundingBox(
        xMin=float(-NODE_MATCH_TOLERANCE), 
        xMax=float(CUBE_SIZE+NODE_MATCH_TOLERANCE),
        yMin=float(-NODE_MATCH_TOLERANCE), 
        yMax=float(CUBE_SIZE+NODE_MATCH_TOLERANCE),
        zMin=float(-NODE_MATCH_TOLERANCE), 
        zMax=float(+NODE_MATCH_TOLERANCE))
    assembly.Set(nodes=bottom_nodes, name='BOTTOM_FACE_NODES')
    print(f"Created TOP_FACE_NODES set with {len(bottom_nodes)} nodes for flux measurement")
    
    # Create a surface for the top face
    bottom_face = instance.faces.getByBoundingBox(
        xMin=float(-NODE_MATCH_TOLERANCE), 
        xMax=float(CUBE_SIZE+NODE_MATCH_TOLERANCE),
        yMin=float(-NODE_MATCH_TOLERANCE), 
        yMax=float(CUBE_SIZE+NODE_MATCH_TOLERANCE),
        zMin=float(-NODE_MATCH_TOLERANCE), 
        zMax=float(+NODE_MATCH_TOLERANCE))
    assembly.Surface(side1Faces=bottom_face, name='BOTTOM_FACE_SURFACE')
    print(f"Created TOP_FACE_SURFACE with {len(bottom_face)} faces for flux measurement")

# ======= FLUX VISUALIZATION  =======

'''def extract_flux_data(odb_path):
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
            if 'MFL' in frame.fieldOutputs:
                flux_field = frame.fieldOutputs['MFL']
                
                # Get flux values for the top face
                top_flux = flux_field.getSubset(region=top_face_set)
                
                # We're interested in the z-component of flux (index 2) at the top face
                z_flux_values = [value.data[2] for value in top_flux.values]
                
                # Calculate average flux
                avg_flux = sum(z_flux_values) / len(z_flux_values)
                avg_flux_values.append(avg_flux)
            else:
                print(f"Warning: MFL not found in frame {frame_idx}")
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
'''

# ====== MAIN MODEL CREATION ======
def create_voxelized_diffusion_model():
    """Create the voxelized cuboid model for diffusion analysis with spherical inclusions"""
    # Start a new model
    model = mdb.Model(name='VoxelizedDiffusionModel')
    if 'Model-1' in mdb.models.keys():
        del mdb.models['Model-1']
    
    # Create the main cuboid part
    s = model.ConstrainedSketch(name='cuboid_sketch', sheetSize=200.0)
    s.rectangle(point1=(0.0, 0.0), point2=(CUBE_SIZE, CUBE_SIZE))
    cuboid_part = model.Part(name='Cuboid', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    cuboid_part.BaseSolidExtrude(sketch=s, depth=CUBE_SIZE)
   
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
                            maxNumInc=100000,
                            minInc=float(MIN_TIME_INCREMENT),
                            maxInc=float(MAX_TIME_INCREMENT),
                            dcmax=float(DCMAX),
                            end=STEADY_STATE_THRESHOLD)  # Maximum concentration change per increment
    
    # Apply face sets for boundary conditions
    #create_face_sets(assembly, instance_name)
    
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
        xMax=float(CUBE_SIZE+0.1), yMax=float(CUBE_SIZE+0.1), zMax=float(0.1))
    assembly.Set(faces=bottom_face, name='Bottom_Face')
    model.ConcentrationBC(name='Bottom_Concentration', 
                         createStepName='DiffusionStep',
                         region=assembly.sets['Bottom_Face'],
                         magnitude=float(BOTTOM_CONCENTRATION),
                         distributionType=UNIFORM)
    
    # Top face with concentration = TOP_CONCENTRATION
    top_face = assembly.instances[instance_name].faces.getByBoundingBox(
        xMin=float(-0.1), yMin=float(-0.1), zMin=float(CUBE_SIZE-0.1),
        xMax=float(CUBE_SIZE+0.1), yMax=float(CUBE_SIZE+0.1), zMax=float(CUBE_SIZE+0.1))
    assembly.Set(faces=top_face, name='Top_Face')
    model.ConcentrationBC(name='Top_Concentration', 
                         createStepName='DiffusionStep',
                         region=assembly.sets['Top_Face'],
                         magnitude=float(TOP_CONCENTRATION),
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
                     magnitudes=(float(0.0),))  
    
    # Create node sets for flux measurement
    create_additional_node_sets(model, assembly, instance_name)
    
    # Add field output request for flux (instead of history output)
    create_output_requests(model)

    # Save the model
    mdb.saveAs(OUTPUT_CAE_PATH)
    print("Diffusion model with periodic boundary conditions created successfully!")
    
    # Run the job and create a plot if successful
    try:
        # Create and submit the job
        job_name = 'VoxelizedDiffusionModel_Job'
        myJob = mdb.Job(name=job_name, model=model.name, description='Diffusion analysis with flux extraction')
        #print(f"Submitting job: {job_name}")
        #myJob.submit()
        #session.jobManager.showMonitor()
        #myJob.waitForCompletion()
        
        # Check if job completed successfully
        '''if myJob.status == COMPLETED and False:
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
            print(f"Job did not complete successfully. Status: {mdb.jobs[job_name].status}")'''
    except Exception as e:
        print(f"Error running job or creating flux plot: {str(e)}")
        traceback.print_exc()
    

# Execute the main function when script is run
if __name__ == "__main__":
    try:
        create_voxelized_diffusion_model()
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        traceback.print_exc()
