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
STEADY_STATE_THRESHOLD = 0.0001 # Threshold for steady-state detection (fraction of initial concentration)

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

def assign_elements_to_inclusions(part, inclusions):
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
    print("\n=== Setting up model ===")
    model = mdb.Model(name='VoxelizedDiffusionModel')
    if 'Model-1' in mdb.models.keys():
        del mdb.models['Model-1']
    
    # Create the main cuboid part
    s = model.ConstrainedSketch(name='cuboid_sketch', sheetSize=200.0)
    s.rectangle(point1=(0.0, 0.0), point2=(CUBE_SIZE, CUBE_SIZE))
    cuboid_part = model.Part(name='Cuboid', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    cuboid_part.BaseSolidExtrude(sketch=s, depth=CUBE_SIZE)
   
    # Read inclusion data
    print("\n=== Reading Input Data ===")
    inclusions, radii = read_inclusion_data(CSV_FILE_PATH)
    print(f"Loaded {len(inclusions)} inclusions from CSV with {len(radii)} unique radii")
        
    # Create materials and sections
    print("\n=== Creating and Assigning Materials ===")
    create_materials(model, radii)
    
    # Assign matrix material to the entire part before meshing
    # Create a set for the entire part (matrix material)
    all_cells = cuboid_part.cells
    cuboid_part.Set(cells=all_cells, name='Matrix_Set')
    cuboid_part.SectionAssignment(region=cuboid_part.sets['Matrix_Set'], sectionName='Matrix_Section')
    
    print("\n=== Meshing ===")
    # Create voxelized mesh
    elem_type = mesh.ElemType(elemCode=DC3D8)
    cuboid_part.setElementType(regions=(cuboid_part.cells,), elemTypes=(elem_type,))
    cuboid_part.seedPart(size=VOXEL_SIZE)
    cuboid_part.generateMesh()
    
    print("\n=== Setting up Assembly ===")
    # Create assembly
    assembly = model.rootAssembly
    instance_name = 'Cuboid-1'
    assembly.Instance(name=instance_name, part=cuboid_part, dependent=ON)
    
    # Assign elements to inclusions
    assign_elements_to_inclusions(cuboid_part, inclusions)
    
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
    print("\n=== Creating Periodic Boundary Conditions ===")
    total_constraints = create_film_periodic_constraints(model, assembly, instance_name)
    print(f"Total periodic constraints created: {total_constraints}")
    
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
