"""
Addition to Abaqus Python script to extract and plot flux data through the end side of the cuboid over time.
This can be added to the existing VoxelApproach.py script.
"""

# ======= FLUX VISUALIZATION ADDITIONS =======
# Add these import statements at the top of your script if not already there
import matplotlib.pyplot as plt
import numpy as np
from odbAccess import *
from abaqusConstants import *

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
def create_additional_node_sets(model, assembly, instance_name):
    """Create additional node sets needed for flux extraction"""
    instance = assembly.instances[instance_name]
    
    # Create top face node set specifically for output
    top_nodes = instance.nodes.getByBoundingBox(
        xMin=float(-NODE_MATCH_TOLERANCE), xMax=float(MODEL_LENGTH+NODE_MATCH_TOLERANCE),
        yMin=float(-NODE_MATCH_TOLERANCE), yMax=float(MODEL_WIDTH+NODE_MATCH_TOLERANCE),
        zMin=float(MODEL_HEIGHT-NODE_MATCH_TOLERANCE), zMax=float(MODEL_HEIGHT+NODE_MATCH_TOLERANCE))
    
    # Create the node set
    assembly.Set(nodes=top_nodes, name='TOP_FACE_NODES')
    print(f"Created TOP_FACE_NODES set with {len(top_nodes)} nodes for flux measurement")

# ===== ADD HISTORY OUTPUT REQUEST =====
def add_flux_history_output(model):
    """Add a history output request for flux data"""
    model.HistoryOutputRequest(name='H-Output-Flux',
                              createStepName='DiffusionStep',
                              variables=('NJFLUX',),
                              region=model.rootAssembly.sets['TOP_FACE_NODES'],
                              sectionPoints=DEFAULT,
                              rebar=EXCLUDE)

# ===== MODIFICATIONS TO MAIN FUNCTION =====
# Add these lines to your create_voxelized_diffusion_model function:
"""
# After applying boundary conditions, add:
create_additional_node_sets(model, assembly, instance_name)
add_flux_history_output(model)

# After running the job (you'll need to add job creation and execution):
model_name = 'VoxelizedDiffusionModel'
job_name = f"{model_name}_Job"
odb_path = f"{job_name}.odb"

# Create and submit the job (add this to your main function)
mdb.Job(name=job_name, model=model_name, description='Diffusion analysis with flux extraction')
print(f"Submitting job: {job_name}")
mdb.jobs[job_name].submit()
mdb.jobs[job_name].waitForCompletion()

# Extract flux data and create plot
time_points, flux_values = extract_flux_data(odb_path)
create_flux_plot(time_points, flux_values, f"{WORKING_DIRECTORY}/flux_vs_time.png")
"""

# ===== UPDATED MAIN FUNCTION WITH JOB EXECUTION =====
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
                           timePeriod=float(TOTAL_TIME), 
                           maxNumInc=100000,
                           initialInc=float(INITIAL_TIME_INCREMENT),
                           minInc=float(MIN_TIME_INCREMENT),
                           maxInc=float(MAX_TIME_INCREMENT),
                           dcmax=float(DCMAX))  # Maximum concentration change per increment
    
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