# -*- coding: utf-8 -*-
import os
import shutil
import csv
import mesh
import csv
import animation
import step
import regionToolset
import time
import regionToolset
import xyPlot
import numpy as np
from abaqus import *
from abaqusConstants import *
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
from abaqus import *
from abaqusConstants import *
from datetime import datetime
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *

'''
#########################################################
#########################################################
#########################################################

                                   **************  WICHTIGE INFORMATIONEN  **************


Dieses Skript wurde von mir, Clemens Koerver, geschrieben.
Kontaktdaten: +49 157/5142406 clemens.koerver@rwth-aachen.de
Bei Fragen gerne melden.
Noah Mentges (Struktursimulation) kennst sich mit dem Thema auch sehr gut aus und hilft sehr gern.


Funktion des Skripts:
Dieses Skript erstellt ein RVEs, das mit Poren aus einer csv Datei gefuellt wird. Randbesingungen und alles weitere werden automatisch erstellt.
Die Constraints werden durch ein Skript (Black Box) von Noah Mentges erstellt. Die csv Poren Daten muessen dazu in dem Ordner "Poren Listen" gespeichert werden.
Der Ordner in dem sie liegen sollten heisst "Poren Listen", alle csv Dateien, die hier abgelegt werden, werden automatisch simuliert.
Das Skript muss direkt in Abaqus ueber File > Run Script... ausgefuehrt werden. Die Ergebnisse werden Automatisch in einem neuen Ordner in "Ergebnisse" gespeichert, das Modell wird automatisch geloescht!


Einheiten:
Dieses Skript nimmt alle Variablen im Nanometer system auf, nicht wie ueblich in Millimetern, weil Zeichnungen nicht so klein erstellt werden koennen.


Folgende Parameter koennen angepasst werden:

Parameter:                      Zeile:                         Funktion:

Material und bauteil Einstellungen:
Diffusionskoeffizient           191                            Passt die Gewschwindigkeit der Diffusion an im Wuerfel
Loeslichkeit                     199.u.362                      Passt die Loeslichkeit in dem Material an
Wuerfel Verschieben              269                            Bestimmt wie der Wuerfel verschoben wird, muss angepasst werden, wenn die Wuerfel Dimesnionen angepasst werden
Diffusionskoeffizient Poren     358                            Diffusionskoeffizienten der Poren, abhaengig von deren Radius

RB und Zeiten:
Mass Diffussion Step (Zeiten)   432-442                        Hier werden die Zeiten der Simulation in [s] eingestellt, diese Einstellungen sind extrem WICHTIG fuer eine fehlerfreie Simulation
Inlet Face                      447                            Definieren der Flaeche fuer Konzentrationsbeaufschlagung, muss angepasst werden, wenn Wuerfel Dimensionen angepasst werden
Concentration magnitude Inlet   462                            Bestimmt die Konzentartion, die auf die Oberseite des Wuerfels aufgegeben wird
Outlet Face                     468                            Definieren der Flaeche fuer Konzentrationsbeaufschlagung, muss angepasst werden, wenn Wuerfel Dimensionen angepasst werden
Concentration magnitude Oulet   481                            Bestimmt die Konzentartion, die auf die Unterseite des Wuerfels aufgegeben wird 

Mesh Einstellungen:
Mesh Controlls (Elemente)       542-555                        Hier werden die Elementtypen und Art der Elementbildung eingestellt
Seed Controlls                  565                            Wichtigste Einstellungen fuer ein Erfolgreiches Mesh UNBEDINGT anpassen, wenn das 0 Elemente erstellt werden

Constraint Einstellungen:
DOF Concentration               583                            hier wird der DOF fuer die Constraints definiert, 11 sollte der Konzentration entsprechen
Node Set                        1245 ff.                       Hier werden die Knoten Abgerufen, die durch Constraints definiert werden muessen, muss ggf. angepasst werden, wenn die Wuerfeldimensionen sich aendern
LatticeVec                      1263                           Definieren der vorgegebenen Abstaende von Knoten und Partnerknoten
Scale Factor                    1265                           Definieren der raeumlichen Toleranz, bei der Suche des Partnerknotens


Die Folgenden Funktionen sind noch zu implementieren:

    - Partitionieren nur der Oberflaechen des Parts Combined Geometry, um ein gleichmaessiges Mesh zu generieren,
    in dem Sicher die Constraints erstellt werden koennen. (Zeile:XXX-XXX)

    - ueberpruefen der Simulationsumgebung, um Simulieren ueber das erste Element hinaus zu ermoeglichen,
    Vielleicht liegt es an den Einheiten (wennn man die Diffusivity sehr hoch stellt funktioniert die Imulation)...
    
    - AM ENDE Modell Loeschen, um Abaqus zu entlasten wieder entkommentieren (Zeile 1333), ggf. odb-Datei schliessung implementieren
    
    - ueberpruefen, ob das Herrausragen einer Pore zu Problemen fuehrt (Ober und Unterseite)
    - Jeder Porenradius solllte nur ein Part mit einem material haben, dass dann  mehrere Instancen generiert
    - Definierte Seedgroesse fuer jede Pore einstellbar machen / alternativ Pore zu Polyeder reduzieren
    
#########################################################
#########################################################
#########################################################
'''


def setup_abaqus_environment(csv_folder, csv_file):
    filename = os.path.join(csv_folder, csv_file)
    try:
        simulation_id = csv_file.split('list_')[1].split('_')[0]  # Nimmt den Teil nach 'list_' bis zum naechsten '_'
    except IndexError:
        print(f"Error with the extraxtion of {csv_file}")
        return False

    ## Wird spaeter verwendet, um die Zeit der Simulation auszugeben
    start_time = time.time()

    #Benennung der Ergebnissdatei mit den uebergebenen Zahlen
    job_name = f"MassDiffusionJob_{simulation_id}"
    odb_file = f"{job_name}.odb"

    print(f"Starting simulation {simulation_counter} with file {csv_file} and naming it {odb_file}")
    
    # Liste mit Poren einlesen
    inclusion_list = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        next(reader)  # ueberspringe die Kopfzeile
        for row in reader:
            row = [float(item) for item in row]
            inclusion_list.append(row)
    inclusion_list = [[3.37, 2.93, 0.69, 1.15, 1.23]]
    print(f'{inclusion_list=}')
    
    #Abaqus Model erstellen
    ModelName = f"RVE_Foam_{simulation_id}"
    mdb.Model(name=ModelName, modelType=STANDARD_EXPLICIT)
    m = mdb.models[ModelName]
    
    
    #### In diesem Abschnitt wird unter anderem die Dauer der Simulation bestimmt, diese Einstellungen sind entscheidend fuer eine gute Simulation ####

    #Diffusion Step fuer die Simulation wird hier erstellt
    m.MassDiffusionStep(
        name='DiffusionStep',  #Name des Schrittes
        previous='Initial',
        timePeriod=10000.0,  #Ganze Simulationszeit
        timeIncrementationMethod=FIXED, #Bestimmt den Modus der Inkremente, kann auch auf Auto gestellt werden (fuehrt aber zu problemen)
        #dcmax=0.5,  #A Float specifying the maximum normalized concentration change to be allowed in an increment. The default value is 0.0.
        initialInc=100,  #Initiales Zeit Inkrement
        minInc=100,
        maxInc=1000
    )

    print("Mass Diffusion Step was created.")

    return_dict = {'simulation_id': simulation_id,
                   'ModelName': ModelName,
                   'm': m,
                   'job_name': job_name,
                   'odb_file': odb_file,
                   'inclusion_list': inclusion_list,
                   'start_time': start_time}
    
    return return_dict

def define_cube_dimensions(setup_data):
    ###### Im Folgenden Abschnitt wird der Kubus erstellt ########

    # Erstellen einer Liste mit allen Poren
    Inclusions = []
    i = 1
    for inclusion in setup_data['inclusion_list']:
        # print (inclusion)
        x_i = (inclusion[0])
        y_i = (inclusion[1])
        z_i = (inclusion[2])
        radius = (inclusion[4])

        Inclusions.append([i, radius, x_i, y_i, z_i])

        i = i + 1

    # Berechnet die maximalen Koordinaten der Poren
    max_x_i = max(inclusion[2] for inclusion in Inclusions)
    max_y_i = max(inclusion[3] for inclusion in Inclusions)
    max_z_i = max(inclusion[4] for inclusion in Inclusions)
    max_radius = max(inclusion[1] for inclusion in Inclusions)
    #print(max_x_i, max_y_i, max_z_i)
    #print(max(max_x_i, max_y_i, max_z_i))
    max_corner = max(max_x_i, max_y_i, max_z_i)

    # Erweitert die Kubus Dimensionen so, dass keine Pore, die Seitenwaende schneidet, die hoehe soll konstant bleiben
    lateral_cube_size = max_corner + 2 * max_radius

    print(f"Cube with a lateral size of {lateral_cube_size} has been created.")
    
    return Inclusions, lateral_cube_size, max_radius

# Tatsaechliches erstellen des Kubus und direktes uebergeben von Materialdaten
def define_cube_material(m, material_name):
    m.Material(name=material_name)

    m.materials[material_name].Elastic(
        table=((200000.0, 0.3),)  # Diese materialdaten sind fuer die Simulation unerheblich
    )

    m.materials[material_name].Diffusivity(
        table=((1.3 * 1e-3,),)  #Diffusionskoeffizient, der die Geschwindigkeit bestimmt Einheit: [nm^2/s]
    )

    m.materials[material_name].Solubility(
        table=((1.0 * 1e-18,),)  #Loeslichkeit Einheit: [
    )

    m.materials[material_name].Density(
        table=((1.0 * 1e-27,),)  #Dichte [kg/nm^3]
    )
    
def create_cube_part(m, lateral_cube_size, max_radius):
    #Zeichnung des Wuerfel und Extrudieren
    m.ConstrainedSketch(name='__profile__', sheetSize=1)
    m.sketches['__profile__'].rectangle(
        point1=(0.0, 0.0),
        point2=(lateral_cube_size, thickness)
    )

    m.Part(
        dimensionality=THREE_D,
        name='Cube',
        type=DEFORMABLE_BODY
    )
    PC = m.parts['Cube']
    PC.BaseSolidExtrude(
        depth=lateral_cube_size,
        sketch=m.sketches['__profile__']
    )
    del m.sketches['__profile__']

    #Wuerfel Material definieren
    define_cube_material(m, cube_material_name)
    # define section
    m.HomogeneousSolidSection(name=cube_section_name, material=cube_material_name, thickness=None)

    #Region erstellen, dem die Materialdaten zugewiesen werden koennen
    region = PC.Set(cells=PC.cells, name='CubeRegion')  # Create a region using a set of cells

    PC.SectionAssignment(
        sectionName=cube_section_name,
        region=region,  # Provide the Region object here
        offset=0.0,
        offsetType=MIDDLE_SURFACE,
        thicknessAssignment=FROM_SECTION
    )

    print('Cube material and Section were assigned.')


    #Passt die Position des Wuerfels so an, dass der Koordinatenursprung wieder an der richtigen Stelle ist
    m.rootAssembly.DatumCsysByDefault(CARTESIAN)
    m.rootAssembly.Instance(
        dependent=OFF,
        name='Cube-1',
        part=PC
    )

    m.rootAssembly.translate(
        instanceList=('Cube-1',),
        vector=(-max_radius, 0, -max_radius)
    )

#Funktion zu Erstellungvon poren
def create_spherical_part(m, partname, i, radius):
    PName = partname + str(i)

    m.ConstrainedSketch(name='__profile__', sheetSize=200.0)
    m.sketches['__profile__'].ConstructionLine(point1=(0.0,
                                                        -100.0), point2=(0.0, 100.0))

    m.sketches['__profile__'].Line(
        point1=(0.0, radius), 
        point2=(0.0, -radius))
    
    m.sketches['__profile__'].ArcByCenterEnds(
        center=(0.0, 0.0),
        direction=CLOCKWISE,
        point1=(0.0, radius),
        point2=(0.0, -radius)
    )

    PE = m.Part(
        dimensionality=THREE_D,
        name=PName,
        type=DEFORMABLE_BODY
    )
    PE.BaseSolidRevolve(
        angle=360.0,
        flipRevolveDirection=OFF,
        sketch=m.sketches['__profile__'])
    del m.sketches['__profile__']

    PE.DatumPlaneByPrincipalPlane(
        offset=0.0,
        principalPlane=XYPLANE
    )
    PE.DatumPlaneByPrincipalPlane(
        offset=0.0,
        principalPlane=YZPLANE
    )
    PE.DatumPlaneByPrincipalPlane(
        offset=0.0,
        principalPlane=XZPLANE
    )

    PE.PartitionCellByDatumPlane(
        cells=PE.cells.findAt(((0.0, 0.0, 0.0),)),
        datumPlane=PE.datums[2]
    )

    PE.PartitionCellByDatumPlane(
        cells=PE.cells.findAt(
            ((0.0, 0.0, 0.01),),
            ((0.0, 0.0, -0.01),),
        ),
        datumPlane=PE.datums[3]
    )

    PE.PartitionCellByDatumPlane(
        cells=PE.cells.findAt(
            ((0.01, 0.0, 0.01),),
            ((-0.01, 0.0, -0.01),),
            ((0.01, 0.0, -0.01),),
            ((-0.01, 0.0, 0.01),),
        ),
        datumPlane=PE.datums[4]
    )

    #print('Spherical part: ' + str(i))
    return PE

def init_pores(m, Inclusions):
        #Erstellen der Poren und zuweisen von Groessenabhaengigen Materialdaten
    instances_list = []
    material_sections = []  # Store material and section assignments

    for Inclusion in Inclusions:
        i = Inclusion[0]
        radius = Inclusion[1]
        x_i = Inclusion[2]
        y_i = Inclusion[3]
        z_i = Inclusion[4]

        # Create spherical part
        PI_i = create_spherical_part(m, 'PI_', i, radius)

        # Create unique material for each inclusion if needed
        material_name = 'Inclusion' + str(i)
        m.Material(name=material_name)

        diffusion_coefficient = 296 * radius * 1e9  # Diffusionskoeffizienten der Poren, abhaengig von deren radius
        m.materials[material_name].Diffusivity(
            table=((diffusion_coefficient,),)
        )
        solubility_coefficient = 1.0 * 1e-18
        m.materials[material_name].Solubility(
            table=((solubility_coefficient,),)
        )

        #Section wird fuer die Poren erstellt
        section_name = 'InclusionSection_' + str(i)
        m.HomogeneousSolidSection(name=section_name, material=material_name, thickness=None)

        #Section wird der Pore zugewiesen
        PI_i.SectionAssignment(
            region=(PI_i.cells,),
            sectionName=section_name
        )

        material_sections.append((PI_i, section_name))

        #Erstellen einer Instance der Poren
        mPI = m.rootAssembly.Instance(
            dependent=ON,
            name='Inclusion_' + str(i),
            part=PI_i
        )

        #Verschieben der Poren an ihren Ort in dem Wuerfel
        m.rootAssembly.translate(
            instanceList=('Inclusion_' + str(i),),
            vector=(x_i, y_i, z_i)
        )

        instances_list.append(mPI)

    instances_list = tuple(instances_list)
    return instances_list

def merge_cube_and_pores(m, lateral_cube_size):
    ##### In diesem Abschnitt wird der Wuerfel mit den Poren vereint und die Materialdaten usw. neu zugeordnet ######

    #Verbinden des Wuerfels und der Poren
    m.rootAssembly.InstanceFromBooleanMerge(
        name='CombinedGeometry',
        instances=(m.rootAssembly.instances['Cube-1'],) + instances_list,
        keepIntersections=ON,
        originalInstances=DELETE,
        domain=GEOMETRY
    )

    print("Cube and Inclusions were combined.")

    combined_part = m.parts['CombinedGeometry']

    #Neuzuweisung der Sections und der Materialdaten fuer den Wuerfel #############noetig?
    #define_cube_material(m, cube_material_name)
    #m.HomogeneousSolidSection(name=cube_section_name, material=cube_material_name, thickness=None)
    cube_region = combined_part.Set(
        cells=combined_part.cells.findAt(((lateral_cube_size / 200, thickness / 200, lateral_cube_size / 200),)), name='CubeRegion') #Hier wird etilt, damit keine Pore mit den Materialdaten erwischt wird
    combined_part.SectionAssignment(
        sectionName=cube_section_name,
        region=cube_region,
        offset=0.0,
        offsetType=MIDDLE_SURFACE,
        thicknessAssignment=FROM_SECTION
    )
    return combined_part

def setup_boundary_conditions(m, lateral_cube_size):
    assembly = m.rootAssembly

    #Sicherstellen dass Die Instance erstellt wurde
    if 'CombinedGeometry-1' not in assembly.instances:
        raise ValueError("Instance 'CombinedGeometry-1' not found in the assembly. Available instances: {}".format(
            assembly.instances.keys()))

    #Definieren der Flaeche fuer Konzentrationsbeaufschlagung
    inlet_face = assembly.instances['CombinedGeometry-1'].faces.findAt(
        ((lateral_cube_size / 2,thickness, lateral_cube_size / 2),))

    if not inlet_face:
        raise ValueError("Inlet Face has not been found.")

    # Region fuer die Inlet Face definieren
    inlet_region = assembly.Set(name='InletFaceSet', faces=inlet_face)

    #Boundary Condition fuer die Inlate face erstellen
    m.ConcentrationBC(
        name='InletConcentration',
        createStepName='DiffusionStep',
        region=inlet_region,
        magnitude=21.0  # Concentration magnitude
    )

    #Finden Der Outlet Face
    outlet_face = assembly.instances['CombinedGeometry-1'].faces.findAt(
        ((lateral_cube_size / 2,0, lateral_cube_size / 2),))

    if not outlet_face:
        raise ValueError("Outlet Face has not been found.")

    #Erstellen einer regionfuer die Oulet face
    outlet_region = assembly.Set(name='OutletFaceSet', faces=outlet_face)

    #Erstellen der Randbesingung fuer die Oulet Face
    m.ConcentrationBC(
        name='OutletConcentration',
        createStepName='DiffusionStep',
        region=outlet_region,
        magnitude=0.0  # Concentration magnitude
    )
    return assembly

def generate_mesh(combined_part):
    ######
    # Im Folgenden wird das Bauteil gemeshed, es ist besonders wichtig in der Command Leiste von Abaqus darauf zu achten,
    # dass ein Mesh mit XXXX Elementen erstellt wurde, da sonst unverstaendlicher Fehler ausgegeben wird,
    # wenn das passiert muessen die Meshparameter ueberarbeitet werden
    #########

    #Es werden alle Teile des Bauteils abgerufen
    cells = combined_part.cells

    #Hier wird die Art der Mesh Elemente und die Bildung der Elemente bestimmt (sollte schon recht optimal gewaehlt sein, wegen komplexer geometrie)
    combined_part.setMeshControls(
        regions=cells,  # Pass the cells directly
        elemShape=TET,  # Tetrahedral elements for complex geometries
        technique=FREE,  # Free meshing technique
        allowMapped=False,  # Disallow mapped meshing since geometry is complex
        sizeGrowth=MODERATE
    )

    # Set element type for the region
    elemType1 = mesh.ElemType(elemCode=DC3D4, elemLibrary=STANDARD)  # Diffusion-capable tetrahedral element
    combined_part.setElementType(
        regions=(cells,),  # Pass the cells as a tuple
        elemTypes=(elemType1,)
    )

    # Seed the part for meshing
    # Seeds werden nur auf Aussenflaechen platziert -> BooleanMerge entfernt alle 'inneren' Flaechen der Poren
    combined_part.seedPart(size=0.3, deviationFactor=0.01, minSizeFactor=0.01)

    print("Mesh configuration was completed.")

    # Generate the mesh for the combined geometry
    combined_part.generateMesh()

    # Access the part
    #combined_part = m.parts['CombinedGeometry']

    # Get the total number of nodes in the mesh
    number_of_nodes = len(combined_part.nodes)

    # Print the number of nodes
    print(f"Die gesamte Anzahl an Knoten in dem Mesh betraegt: {number_of_nodes}")

    ########
    

#Der folgende Abschnitt ist eine BLACK BOX von Noah Mentges! Sie erstellt die Constraints (auch periodische Randbedingungen genannt).
#Dazu wird das Modell und ein Nodeset uebergeben. Das Nodeset enthaelt alle Knoten, denen ein Constraint zugeordnet werden soll.
#Fuer jeden Knoten wird dann ein Partner Knoten gesucht, der nur in den durch den LatticeVec definierten Abstaenden liegen darf.
#Der Partnerknoten, darf dabei leicht abweichen, diese Toleranz wird durch den scale Faktor definiert. Sie sollte nicht zu gross eingestellt werden, um Fehler zu vermeiden!
#Wird ein Knoten gefunden, wird eine Constraint mit dem unten definierten DOF erstellt.
##########

def PeriodicBound3D(mdb, NameModel, NameSet, lateral_cube_size):
    #Definieren der vorgegebenen Abstaende von Knoten und Partnerknoten
    LatticeVec = [[lateral_cube_size, 0, 0], [0, 0, lateral_cube_size]]
    #Definieren der raeumlichen Toleranz, bei der Suche des Partnerknotens, sollte ca. der Elementgroesse entsprechen, aber nicht viel groesser!
    scale = 2
    import time
    start1 = time.time()

    # Freiheitsgrad Konzentration
    DOF_Conc = 11

    from part import THREE_D, DEFORMABLE_BODY
    # Create reference parts and assemble
    # NameRef1='RefPoint-0'; NameRef2='RefPoint-1'; NameRef3='RefPoint-2'
    # mdb.models[NameModel].Part(dimensionality=THREE_D, name=NameRef1, type=
    #     DEFORMABLE_BODY)
    # mdb.models[NameModel].parts[NameRef1].ReferencePoint(point=(LatticeVec[0][0], 0.0, LatticeVec[2][2]/2.))
    # mdb.models[NameModel].Part(dimensionality=THREE_D, name=NameRef2, type=
    #     DEFORMABLE_BODY)
    # mdb.models[NameModel].parts[NameRef2].ReferencePoint(point=(0.0, LatticeVec[1][1], LatticeVec[2][2]/2.))
    # mdb.models[NameModel].Part(dimensionality=THREE_D, name=NameRef3, type=
    #     DEFORMABLE_BODY)
    # mdb.models[NameModel].parts[NameRef3].ReferencePoint(point=(0.0, 0.0, LatticeVec[2][2] * 1.5))
    # mdb.models[NameModel].rootAssembly.Instance(dependent=ON, name=NameRef1,
    #     part=mdb.models[NameModel].parts[NameRef1])
    # mdb.models[NameModel].rootAssembly.Instance(dependent=ON, name=NameRef2,
    #     part=mdb.models[NameModel].parts[NameRef2])
    # mdb.models[NameModel].rootAssembly.Instance(dependent=ON, name=NameRef3,
    #     part=mdb.models[NameModel].parts[NameRef3])

    # Create set of reference points
    # mdb.models[NameModel].rootAssembly.Set(name=NameRef1, referencePoints=(
    #     mdb.models[NameModel].rootAssembly.instances[NameRef1].referencePoints[1],))
    # mdb.models[NameModel].rootAssembly.Set(name=NameRef2, referencePoints=(
    #     mdb.models[NameModel].rootAssembly.instances[NameRef2].referencePoints[1],))
    # mdb.models[NameModel].rootAssembly.Set(name=NameRef3, referencePoints=(
    #     mdb.models[NameModel].rootAssembly.instances[NameRef3].referencePoints[1],))
    end1 = time.time()
    #print(end1 - start1)
    start2 = time.time()
    # Get all nodes
    nodesAll = mdb.models[NameModel].parts['CombinedGeometry'].allSets[NameSet].nodes
    nodesAllCoor = []
    # print(nodesAll)
    for nod in mdb.models[NameModel].parts['CombinedGeometry'].allSets[NameSet].nodes:
        nodesAllCoor.append(nod.coordinates)
    # print(nodesAllCoor)
    end2 = time.time()
    #print(end2 - start2)
    start3 = time.time()
    repConst = 0
    # Find periodically located nodes and apply equation constraints
    ranNodes = range(0, len(nodesAll))  # Index array of nodes not used in equations constraint
    #print(len(nodesAll))
    for repnod1 in range(0, len(nodesAll)):
        stop = False  # Stop will become true when equation constraint is made between nodes
        #print(repnod1)
        #print(nodesAllCoor)
        Coor1 = nodesAllCoor[repnod1]  # Coordinates of Node 1
        for repnod2 in ranNodes:  # Loop over all available nodes
            Coor2 = nodesAllCoor[repnod2]  # Coordinates of Node 2
            for comb in range(0, len(LatticeVec)):  # Check if nodes are located exactly the vector lattice apart
                if int(scale * (LatticeVec[comb][0] - Coor2[0] + Coor1[0])) == 0 and int(
                        scale * (LatticeVec[comb][1] - Coor2[1] + Coor1[1])) == 0 and int(
                        scale * (LatticeVec[comb][2] - Coor2[2] + Coor1[2])) == 0:
                    # Correct combination found
                    # Create sets for use in equations constraints
                    mdb.models[NameModel].parts['CombinedGeometry'].Set(name='Node-1-' + str(repConst), nodes=
                    mdb.models[NameModel].parts['CombinedGeometry'].allSets[NameSet].nodes[repnod1:repnod1 + 1])
                    mdb.models[NameModel].parts['CombinedGeometry'].Set(name='Node-2-' + str(repConst), nodes=
                    mdb.models[NameModel].parts['CombinedGeometry'].allSets[NameSet].nodes[repnod2:repnod2 + 1])
                    # Create equations constraints for each dof
                    # for Dim1 in [1,2,3]:
                    #     mdb.models[NameModel].Equation(name='PerConst'+str(Dim1)+'-'+str(repConst),
                    #         terms=((1.0,'CombinedGeometry-1.Node-1-'+str(repConst), Dim1),(-1.0, 'CombinedGeometry-1.Node-2-'+str(repConst), Dim1) ,
                    #             (1.0, 'RefPoint-'+str(comb), Dim1)))
                    mdb.models[NameModel].Equation(name='PerConst' + str(DOF_Conc) + '-' + str(repConst),
                                                    terms=((1.0, 'CombinedGeometry-1.Node-1-' + str(repConst), DOF_Conc),
                                                            (-1.0, 'CombinedGeometry-1.Node-2-' + str(repConst), DOF_Conc)))
                    repConst = repConst + 1  # Increase integer for naming equation constraint
                    # ranNodes.remove(repnod1)        #Remove used node from available list
                    stop = True  # Don't look further, go to following node.
                    break
            if stop:
                break
    end3 = time.time()
    #print(end3 - start3)
    # Return coordinates of free node so that it can be fixed
    if ranNodes == []:
        return (nodesAll[0].coordinates, NameRef1, NameRef2, NameRef3)
    # return (nodesAll[ranNodes[0]].coordinates, NameRef1, NameRef2, NameRef3)
    return (nodesAll[ranNodes[0]].coordinates)

# unused
def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)

# unused
def delete_duplicate_nodes(mdb, NameModel, NameSet, pref_inst_list, ddTolerance=1.e-3):
    nodeSet = mdb.models[NameModel].rootAssembly.sets[NameSet]
    sorted = [[node.coordinates[0], node.coordinates[1], node.coordinates[2], node.label, node.instanceName] for
                node in nodeSet.nodes]
    sorted.sort()

    jump = False
    jump_next = False
    identifiers = 0
    for i in range(len(sorted) - 1):
        if not jump_next:
            if (abs(sorted[i][0] - sorted[i + 1][0]) < ddTolerance) and (
                    abs(sorted[i][1] - sorted[i + 1][1]) < ddTolerance) and (
                    abs(sorted[i][2] - sorted[i + 1][2]) < ddTolerance):
                for pref in pref_inst_list:
                    if sorted[i + 1][4] == pref:
                        jump = True
                        break
                    elif sorted[i] == pref:
                        jump_next = True
                        break
            if not jump:
                if identifiers == 0:
                    identifiers = [[sorted[i][4], sorted[i][3]]]
                else:
                    identifiers.append([sorted[i][4], sorted[i][3]])
            else:
                print(sorted[i])
                jump = False
        else:
            print(sorted[i])
            jump_next = False
    # print(identifiers)

    sorted_ids = 0

    for id in identifiers:
        if sorted_ids == 0:
            sorted_ids = [[id[0], [id[1]]]]
            # print(sorted_ids)
        else:
            new_item = True
            for j, inst in enumerate(sorted_ids):
                # print(inst)
                # print(id)
                if inst[0] == id[0]:
                    sorted_ids[j][1].append(id[1])
                    new_item = False
            if new_item:
                sorted_ids.append([id[0], [id[1]]])

    set_tuple = to_tuple(sorted_ids)

    mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name=NameSet, nodeLabels=set_tuple)

# unused
def delete_slave_nodes_from_set(mdb, NameModel, NameSet, slave_nodes):
    nodeSet = mdb.models[NameModel].rootAssembly.sets[NameSet]
    node_list = list(nodeSet.nodes)
    slave_node_list = slave_nodes

    new_node_list = 0
    for node in node_list:
        if not node in slave_node_list:
            if new_node_list == 0:
                new_node_list = [node]
            else:
                new_node_list.append(node)

    sorted_ids = 0

    for node in new_node_list:
        id = [node.instanceName, node.label]
        if sorted_ids == 0:
            sorted_ids = [[id[0], [id[1]]]]
            # print(sorted_ids)
        else:
            new_item = True
            for j, inst in enumerate(sorted_ids):
                # print(inst)
                # print(id)
                if inst[0] == id[0]:
                    sorted_ids[j][1].append(id[1])
                    new_item = False
                    break
            if new_item:
                sorted_ids.append([id[0], [id[1]]])

    set_tuple = to_tuple(sorted_ids)

    mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name=NameSet, nodeLabels=set_tuple)


#pbc_tolerance = 1e3
# unused
def create_PBC_sorted(mdb, NameModel, NameSet, box_tol=1.e-3, RP_Names=['RefPoint-0', 'RefPoint-1', 'RefPoint-2'],
                        delete_duplicates=False, duplicate_pref_list=[''], ddTolerance=1.e-3):

    """
    Creates periodic boundary conditions by sorting nodes by coordinates. Periodic meshes are necessary.

    Input Variables:

    mdb:        Model database object
    NameModel:  String with abaqus-model name
    NameSet:    String with name of set with surface nodes
    RVE_Dims:   List with RVE-dimensions in form [x, y, z]
    ref_coords: Vector to most negative point (Default: [0, 0, 0])
    box_tol:    Tolerance for size of bounding boxes (usually not necessary to vary)
    """
    import time
    m = mdb.models[NameModel]

    if delete_duplicates:
        delete_duplicate_nodes(mdb, NameModel, NameSet, duplicate_pref_list, ddTolerance)

    bounds = m.rootAssembly.sets[NameSet].nodes.getBoundingBox()

    print(bounds['low'])
    print(bounds['high'])

    RVE_Dims = [bounds['high'][i] - bounds['low'][i] for i in range(3)]

    ref_coords = [bounds['low'][i] for i in range(3)]

    LatticeVec = [[RVE_Dims[0], 0., 0.],
                    [0., RVE_Dims[1], 0.],
                    [0., 0., RVE_Dims[2]]]

    # Create reference parts and assemble
    NameRef1 = RP_Names[0];
    NameRef2 = RP_Names[1];
    NameRef3 = RP_Names[2]
    mdb.models[NameModel].Part(dimensionality=THREE_D, name=NameRef1, type=
    DEFORMABLE_BODY)
    mdb.models[NameModel].parts[NameRef1].ReferencePoint(point=(ref_coords[0] + 1.5 * LatticeVec[0][0],
                                                                ref_coords[1] + LatticeVec[1][1] / 2.,
                                                                ref_coords[2] + LatticeVec[2][2] / 2.))
    mdb.models[NameModel].Part(dimensionality=THREE_D, name=NameRef2, type=
    DEFORMABLE_BODY)
    mdb.models[NameModel].parts[NameRef2].ReferencePoint(point=(ref_coords[0] + LatticeVec[0][0] / 2.,
                                                                ref_coords[1] + 1.5 * LatticeVec[1][1],
                                                                ref_coords[2] + LatticeVec[2][2] / 2.))
    mdb.models[NameModel].Part(dimensionality=THREE_D, name=NameRef3, type=
    DEFORMABLE_BODY)
    mdb.models[NameModel].parts[NameRef3].ReferencePoint(point=(ref_coords[0] + LatticeVec[0][0] / 2.,
                                                                ref_coords[1] + LatticeVec[1][1] / 2.,
                                                                ref_coords[2] + LatticeVec[2][2] * 1.5))
    mdb.models[NameModel].rootAssembly.Instance(dependent=ON, name=NameRef1,
                                                part=mdb.models[NameModel].parts[NameRef1])
    mdb.models[NameModel].rootAssembly.Instance(dependent=ON, name=NameRef2,
                                                part=mdb.models[NameModel].parts[NameRef2])
    mdb.models[NameModel].rootAssembly.Instance(dependent=ON, name=NameRef3,
                                                part=mdb.models[NameModel].parts[NameRef3])

    # Create set of reference points
    mdb.models[NameModel].rootAssembly.Set(name=NameRef1, referencePoints=(
        mdb.models[NameModel].rootAssembly.instances[NameRef1].referencePoints[1],))
    mdb.models[NameModel].rootAssembly.Set(name=NameRef2, referencePoints=(
        mdb.models[NameModel].rootAssembly.instances[NameRef2].referencePoints[1],))
    mdb.models[NameModel].rootAssembly.Set(name=NameRef3, referencePoints=(
        mdb.models[NameModel].rootAssembly.instances[NameRef3].referencePoints[1],))

    start3 = time.time()

    # Get Face Nodes
    x_pos = m.rootAssembly.sets[NameSet].nodes.getByBoundingBox(xMin=ref_coords[0] + LatticeVec[0][0] - box_tol,
                                                                xMax=ref_coords[0] + LatticeVec[0][0] + box_tol,
                                                                yMin=ref_coords[1] + box_tol,
                                                                yMax=ref_coords[1] + LatticeVec[1][1] - box_tol,
                                                                zMin=ref_coords[2] + box_tol,
                                                                zMax=ref_coords[2] + LatticeVec[2][2] - box_tol)
    x_neg = m.rootAssembly.sets[NameSet].nodes.getByBoundingBox(xMin=ref_coords[0] - box_tol,
                                                                xMax=ref_coords[0] + box_tol,
                                                                yMin=ref_coords[1] + box_tol,
                                                                yMax=ref_coords[1] + LatticeVec[1][1] - box_tol,
                                                                zMin=ref_coords[2] + box_tol,
                                                                zMax=ref_coords[2] + LatticeVec[2][2] - box_tol)
    y_pos = m.rootAssembly.sets[NameSet].nodes.getByBoundingBox(xMin=ref_coords[0] + box_tol,
                                                                xMax=ref_coords[0] + LatticeVec[0][0] - box_tol,
                                                                yMin=ref_coords[1] + LatticeVec[1][1] - box_tol,
                                                                yMax=ref_coords[1] + LatticeVec[1][1] + box_tol,
                                                                zMin=ref_coords[2] + box_tol,
                                                                zMax=ref_coords[2] + LatticeVec[2][2] - box_tol)
    y_neg = m.rootAssembly.sets[NameSet].nodes.getByBoundingBox(xMin=ref_coords[0] + box_tol,
                                                                xMax=ref_coords[0] + LatticeVec[0][0] - box_tol,
                                                                yMin=ref_coords[1] - box_tol,
                                                                yMax=ref_coords[1] + box_tol,
                                                                zMin=ref_coords[2] + box_tol,
                                                                zMax=ref_coords[2] + LatticeVec[2][2] - box_tol)
    z_pos = m.rootAssembly.sets[NameSet].nodes.getByBoundingBox(xMin=ref_coords[0] + box_tol,
                                                                xMax=ref_coords[0] + LatticeVec[0][0] - box_tol,
                                                                yMin=ref_coords[1] + box_tol,
                                                                yMax=ref_coords[1] + LatticeVec[1][1] - box_tol,
                                                                zMin=ref_coords[2] + LatticeVec[2][2] - box_tol,
                                                                zMax=ref_coords[2] + LatticeVec[2][2] + box_tol)
    z_neg = m.rootAssembly.sets[NameSet].nodes.getByBoundingBox(xMin=ref_coords[0] + box_tol,
                                                                xMax=ref_coords[0] + LatticeVec[0][0] - box_tol,
                                                                yMin=ref_coords[1] + box_tol,
                                                                yMax=ref_coords[1] + LatticeVec[1][1] - box_tol,
                                                                zMin=ref_coords[2] - box_tol,
                                                                zMax=ref_coords[2] + box_tol)

    # Sort Face Nodes by coordinates
    print("Sorting Face Nodes")
    x_pos_coord = [[node.coordinates[0], node.coordinates[1], node.coordinates[2], node.label, node.instanceName]
                    for node in x_pos]
    x_pos_coord.sort()
    x_neg_coord = [[node.coordinates[0], node.coordinates[1], node.coordinates[2], node.label, node.instanceName]
                    for node in x_neg]
    x_neg_coord.sort()
    y_pos_coord = [[node.coordinates[0], node.coordinates[1], node.coordinates[2], node.label, node.instanceName]
                    for node in y_pos]
    y_pos_coord.sort()
    y_neg_coord = [[node.coordinates[0], node.coordinates[1], node.coordinates[2], node.label, node.instanceName]
                    for node in y_neg]
    y_neg_coord.sort()
    z_pos_coord = [[node.coordinates[0], node.coordinates[1], node.coordinates[2], node.label, node.instanceName]
                    for node in z_pos]
    z_pos_coord.sort()
    z_neg_coord = [[node.coordinates[0], node.coordinates[1], node.coordinates[2], node.label, node.instanceName]
                    for node in z_neg]
    z_neg_coord.sort()

    # Create PBC for each face
    print("Creating PBC for x-faces")
    for i in range(0, len(x_pos_coord)):
        # print(x_pos_coord[i])
        # print(x_neg_coord[i])
        mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-x-pos-' + str(i), nodeLabels=(
        (x_pos_coord[i][4], (x_pos_coord[i][3],)),))
        mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-x-neg-' + str(i), nodeLabels=(
        (x_neg_coord[i][4], (x_neg_coord[i][3],)),))

        for Dim1 in [1, 2, 3]:
            mdb.models[NameModel].Equation(name='PBC_x_Const' + str(Dim1) + '-' + str(i),
                                            terms=(
                                            (1.0, 'PBC-x-pos-' + str(i), Dim1), (-1.0, 'PBC-x-neg-' + str(i), Dim1),
                                            (-1.0, RP_Names[0], Dim1)))

    print("Creating PBC for y-faces")
    for i in range(0, len(y_pos_coord)):
        mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-y-pos-' + str(i), nodeLabels=(
        (y_pos_coord[i][4], (y_pos_coord[i][3],)),))
        mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-y-neg-' + str(i), nodeLabels=(
        (y_neg_coord[i][4], (y_neg_coord[i][3],)),))
        for Dim1 in [1, 2, 3]:
            mdb.models[NameModel].Equation(name='PBC_y_Const' + str(Dim1) + '-' + str(i),
                                            terms=(
                                            (1.0, 'PBC-y-pos-' + str(i), Dim1), (-1.0, 'PBC-y-neg-' + str(i), Dim1),
                                            (-1.0, RP_Names[1], Dim1)))

    print("Creating PBC for z-faces")
    for i in range(0, len(z_pos_coord)):
        mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-z-pos-' + str(i), nodeLabels=(
        (z_pos_coord[i][4], (z_pos_coord[i][3],)),))
        mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-z-neg-' + str(i), nodeLabels=(
        (z_neg_coord[i][4], (z_neg_coord[i][3],)),))
        for Dim1 in [1, 2, 3]:
            mdb.models[NameModel].Equation(name='PBC_z_Const' + str(Dim1) + '-' + str(i),
                                            terms=(
                                            (1.0, 'PBC-z-pos-' + str(i), Dim1), (-1.0, 'PBC-z-neg-' + str(i), Dim1),
                                            (-1.0, RP_Names[2], Dim1)))
    end3 = time.time()

    # Create PBC for each edge
    edge_ab = m.rootAssembly.sets[NameSet].nodes.getByBoundingCylinder(
        (ref_coords[0] + box_tol, ref_coords[1], ref_coords[2]),
        (ref_coords[0] + LatticeVec[0][0] - box_tol, ref_coords[1], ref_coords[2]), box_tol)
    edge_dc = m.rootAssembly.sets[NameSet].nodes.getByBoundingCylinder(
        (ref_coords[0] + box_tol, ref_coords[1] + LatticeVec[1][1], ref_coords[2]),
        (ref_coords[0] + LatticeVec[0][0] - box_tol, ref_coords[1] + LatticeVec[1][1], ref_coords[2]),
        ref_coords[2] + box_tol)
    edge_hg = m.rootAssembly.sets[NameSet].nodes.getByBoundingCylinder(
        (ref_coords[0] + box_tol, ref_coords[1] + LatticeVec[1][1], ref_coords[2] + LatticeVec[2][2]),
        (ref_coords[0] + LatticeVec[0][0] - box_tol, ref_coords[1] + LatticeVec[1][1],
            ref_coords[2] + LatticeVec[2][2]), box_tol)
    edge_ef = m.rootAssembly.sets[NameSet].nodes.getByBoundingCylinder(
        (ref_coords[0] + box_tol, ref_coords[1], ref_coords[2] + LatticeVec[2][2]),
        (ref_coords[0] + LatticeVec[0][0] - box_tol, ref_coords[1], ref_coords[2] + LatticeVec[2][2]), box_tol)

    edge_ab_coord = [[node.coordinates[0], node.coordinates[1], node.coordinates[2], node.label, node.instanceName]
                        for node in edge_ab]
    edge_ab_coord.sort()
    edge_dc_coord = [[node.coordinates[0], node.coordinates[1], node.coordinates[2], node.label, node.instanceName]
                        for node in edge_dc]
    edge_dc_coord.sort()
    edge_hg_coord = [[node.coordinates[0], node.coordinates[1], node.coordinates[2], node.label, node.instanceName]
                        for node in edge_hg]
    edge_hg_coord.sort()
    edge_ef_coord = [[node.coordinates[0], node.coordinates[1], node.coordinates[2], node.label, node.instanceName]
                        for node in edge_ef]
    edge_ef_coord.sort()

    edge_ad = m.rootAssembly.sets[NameSet].nodes.getByBoundingCylinder(
        (ref_coords[0], ref_coords[1] + box_tol, ref_coords[2]),
        (ref_coords[0], ref_coords[1] + LatticeVec[1][1] - box_tol, ref_coords[2]), box_tol)
    edge_bc = m.rootAssembly.sets[NameSet].nodes.getByBoundingCylinder(
        (ref_coords[0] + LatticeVec[0][0], ref_coords[1] + box_tol, ref_coords[2]),
        (ref_coords[0] + LatticeVec[0][0], ref_coords[1] + LatticeVec[1][1] - box_tol, ref_coords[2]), box_tol)
    edge_fg = m.rootAssembly.sets[NameSet].nodes.getByBoundingCylinder(
        (ref_coords[0] + LatticeVec[0][0], ref_coords[1] + box_tol, ref_coords[2] + LatticeVec[2][2]),
        (ref_coords[0] + LatticeVec[0][0], ref_coords[1] + LatticeVec[1][1] - box_tol,
            ref_coords[2] + LatticeVec[2][2]), box_tol)
    edge_eh = m.rootAssembly.sets[NameSet].nodes.getByBoundingCylinder(
        (ref_coords[0], ref_coords[1] + box_tol, ref_coords[2] + LatticeVec[2][2]),
        (ref_coords[0], ref_coords[1] + LatticeVec[1][1] - box_tol, ref_coords[2] + LatticeVec[2][2]), box_tol)

    edge_ad_coord = [[node.coordinates[0], node.coordinates[1], node.coordinates[2], node.label, node.instanceName]
                        for node in edge_ad]
    edge_ad_coord.sort()
    edge_bc_coord = [[node.coordinates[0], node.coordinates[1], node.coordinates[2], node.label, node.instanceName]
                        for node in edge_bc]
    edge_bc_coord.sort()
    edge_fg_coord = [[node.coordinates[0], node.coordinates[1], node.coordinates[2], node.label, node.instanceName]
                        for node in edge_fg]
    edge_fg_coord.sort()
    edge_eh_coord = [[node.coordinates[0], node.coordinates[1], node.coordinates[2], node.label, node.instanceName]
                        for node in edge_eh]
    edge_eh_coord.sort()

    edge_ae = m.rootAssembly.sets[NameSet].nodes.getByBoundingCylinder(
        (ref_coords[0], ref_coords[1], ref_coords[2] + box_tol),
        (ref_coords[0], ref_coords[1], ref_coords[2] + LatticeVec[2][2] - box_tol), box_tol)
    edge_bf = m.rootAssembly.sets[NameSet].nodes.getByBoundingCylinder(
        (ref_coords[0] + LatticeVec[0][0], ref_coords[1], ref_coords[2] + box_tol),
        (ref_coords[0] + LatticeVec[0][0], ref_coords[1], ref_coords[2] + LatticeVec[2][2] - box_tol), box_tol)
    edge_cg = m.rootAssembly.sets[NameSet].nodes.getByBoundingCylinder(
        (ref_coords[0] + LatticeVec[0][0], ref_coords[1] + LatticeVec[1][1], ref_coords[2] + box_tol),
        (ref_coords[0] + LatticeVec[0][0], ref_coords[1] + LatticeVec[1][1],
            ref_coords[2] + LatticeVec[2][2] - box_tol), box_tol)
    edge_dh = m.rootAssembly.sets[NameSet].nodes.getByBoundingCylinder(
        (ref_coords[0], ref_coords[1] + LatticeVec[1][1], ref_coords[2] + box_tol),
        (ref_coords[0], ref_coords[1] + LatticeVec[1][1], ref_coords[2] + LatticeVec[2][2] - box_tol), box_tol)

    edge_ae_coord = [[node.coordinates[0], node.coordinates[1], node.coordinates[2], node.label, node.instanceName]
                        for node in edge_ae]
    edge_ae_coord.sort()
    edge_bf_coord = [[node.coordinates[0], node.coordinates[1], node.coordinates[2], node.label, node.instanceName]
                        for node in edge_bf]
    edge_bf_coord.sort()
    edge_cg_coord = [[node.coordinates[0], node.coordinates[1], node.coordinates[2], node.label, node.instanceName]
                        for node in edge_cg]
    edge_cg_coord.sort()
    edge_dh_coord = [[node.coordinates[0], node.coordinates[1], node.coordinates[2], node.label, node.instanceName]
                        for node in edge_dh]
    edge_dh_coord.sort()

    print("Creating PBC for x-edges")
    for i in range(0, len(edge_bc_coord)):
        mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-edge-bc-' + str(i), nodeLabels=(
        (edge_bc_coord[i][4], (edge_bc_coord[i][3],)),))
        mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-edge-ad-' + str(i), nodeLabels=(
        (edge_ad_coord[i][4], (edge_ad_coord[i][3],)),))
        for Dim1 in [1, 2, 3]:
            mdb.models[NameModel].Equation(name='PBC-bc-ad' + str(Dim1) + '-' + str(i),
                                            terms=((1.0, 'PBC-edge-bc-' + str(i), Dim1),
                                                    (-1.0, 'PBC-edge-ad-' + str(i), Dim1),
                                                    (-1.0, RP_Names[0], Dim1)))
    for i in range(0, len(edge_fg_coord)):
        mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-edge-fg-' + str(i), nodeLabels=(
        (edge_fg_coord[i][4], (edge_fg_coord[i][3],)),))
        mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-edge-eh-' + str(i), nodeLabels=(
        (edge_eh_coord[i][4], (edge_eh_coord[i][3],)),))
        for Dim1 in [1, 2, 3]:
            mdb.models[NameModel].Equation(name='PBC-fg-eh' + str(Dim1) + '-' + str(i),
                                            terms=((1.0, 'PBC-edge-fg-' + str(i), Dim1),
                                                    (-1.0, 'PBC-edge-eh-' + str(i), Dim1),
                                                    (-1.0, RP_Names[0], Dim1)))
    for i in range(0, len(edge_bf_coord)):
        mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-edge-bf-' + str(i), nodeLabels=(
        (edge_bf_coord[i][4], (edge_bf_coord[i][3],)),))
        mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-edge-ae-' + str(i), nodeLabels=(
        (edge_ae_coord[i][4], (edge_ae_coord[i][3],)),))
        for Dim1 in [1, 2, 3]:
            mdb.models[NameModel].Equation(name='PBC-bf-ae' + str(Dim1) + '-' + str(i),
                                            terms=((1.0, 'PBC-edge-bf-' + str(i), Dim1),
                                                    (-1.0, 'PBC-edge-ae-' + str(i), Dim1),
                                                    (-1.0, RP_Names[0], Dim1)))

    print("Creating PBC for y-edges")
    for i in range(0, len(edge_dh_coord)):
        mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-edge-dh-' + str(i), nodeLabels=(
        (edge_dh_coord[i][4], (edge_dh_coord[i][3],)),))
        mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-edge-ae-' + str(i), nodeLabels=(
        (edge_ae_coord[i][4], (edge_ae_coord[i][3],)),))
        for Dim1 in [1, 2, 3]:
            mdb.models[NameModel].Equation(name='PBC-dh-ae' + str(Dim1) + '-' + str(i),
                                            terms=((1.0, 'PBC-edge-dh-' + str(i), Dim1),
                                                    (-1.0, 'PBC-edge-ae-' + str(i), Dim1),
                                                    (-1.0, RP_Names[1], Dim1)))
    for i in range(0, len(edge_hg_coord)):
        mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-edge-hg-' + str(i), nodeLabels=(
        (edge_hg_coord[i][4], (edge_hg_coord[i][3],)),))
        mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-edge-ef-' + str(i), nodeLabels=(
        (edge_ef_coord[i][4], (edge_ef_coord[i][3],)),))
        for Dim1 in [1, 2, 3]:
            mdb.models[NameModel].Equation(name='PBC-hg-ef' + str(Dim1) + '-' + str(i),
                                            terms=((1.0, 'PBC-edge-hg-' + str(i), Dim1),
                                                    (-1.0, 'PBC-edge-ef-' + str(i), Dim1),
                                                    (-1.0, RP_Names[1], Dim1)))
    for i in range(0, len(edge_cg_coord)):
        mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-edge-cg-' + str(i), nodeLabels=(
        (edge_cg_coord[i][4], (edge_cg_coord[i][3],)),))
        mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-edge-bf-' + str(i), nodeLabels=(
        (edge_bf_coord[i][4], (edge_bf_coord[i][3],)),))
        for Dim1 in [1, 2, 3]:
            mdb.models[NameModel].Equation(name='PBC-cg-bf' + str(Dim1) + '-' + str(i),
                                            terms=((1.0, 'PBC-edge-cg-' + str(i), Dim1),
                                                    (-1.0, 'PBC-edge-bf-' + str(i), Dim1),
                                                    (-1.0, RP_Names[1], Dim1)))
    for i in range(0, len(edge_dc_coord)):
        mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-edge-dc-' + str(i), nodeLabels=(
        (edge_dc_coord[i][4], (edge_dc_coord[i][3],)),))
        mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-edge-ab-' + str(i), nodeLabels=(
        (edge_ab_coord[i][4], (edge_ab_coord[i][3],)),))
        for Dim1 in [1, 2, 3]:
            mdb.models[NameModel].Equation(name='PBC-dc-ab' + str(Dim1) + '-' + str(i),
                                            terms=((1.0, 'PBC-edge-dc-' + str(i), Dim1),
                                                    (-1.0, 'PBC-edge-ab-' + str(i), Dim1),
                                                    (-1.0, RP_Names[1], Dim1)))

    print("Creating PBC for z-edges")
    for i in range(0, len(edge_ef_coord)):
        mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-edge-ef-' + str(i), nodeLabels=(
        (edge_ef_coord[i][4], (edge_ef_coord[i][3],)),))
        mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-edge-ab-' + str(i), nodeLabels=(
        (edge_ab_coord[i][4], (edge_ab_coord[i][3],)),))
        for Dim1 in [1, 2, 3]:
            mdb.models[NameModel].Equation(name='PBC-ef-ab' + str(Dim1) + '-' + str(i),
                                            terms=((1.0, 'PBC-edge-ef-' + str(i), Dim1),
                                                    (-1.0, 'PBC-edge-ab-' + str(i), Dim1),
                                                    (-1.0, RP_Names[2], Dim1)))
    for i in range(0, len(edge_eh_coord)):
        mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-edge-eh-' + str(i), nodeLabels=(
        (edge_eh_coord[i][4], (edge_eh_coord[i][3],)),))
        mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-edge-ad-' + str(i), nodeLabels=(
        (edge_ad_coord[i][4], (edge_ad_coord[i][3],)),))
        for Dim1 in [1, 2, 3]:
            mdb.models[NameModel].Equation(name='PBC-eh-ad' + str(Dim1) + '-' + str(i),
                                            terms=((1.0, 'PBC-edge-eh-' + str(i), Dim1),
                                                    (-1.0, 'PBC-edge-ad-' + str(i), Dim1),
                                                    (-1.0, RP_Names[2], Dim1)))

    # Create PBC for vertices
    print("Creating PBC for vertices")

    vertex_a = m.rootAssembly.sets[NameSet].nodes.getByBoundingSphere((ref_coords[0], ref_coords[1], ref_coords[2]),
                                                                        box_tol)
    vertex_b = m.rootAssembly.sets[NameSet].nodes.getByBoundingSphere(
        (ref_coords[0] + LatticeVec[0][0], ref_coords[1], ref_coords[2]), box_tol)
    vertex_c = m.rootAssembly.sets[NameSet].nodes.getByBoundingSphere(
        (ref_coords[0] + LatticeVec[0][0], ref_coords[1] + LatticeVec[1][1], ref_coords[2]), box_tol)
    vertex_d = m.rootAssembly.sets[NameSet].nodes.getByBoundingSphere(
        (ref_coords[0], ref_coords[1] + LatticeVec[1][1], ref_coords[2]), box_tol)
    vertex_e = m.rootAssembly.sets[NameSet].nodes.getByBoundingSphere(
        (ref_coords[0], ref_coords[1], ref_coords[2] + LatticeVec[2][2]), box_tol)
    vertex_f = m.rootAssembly.sets[NameSet].nodes.getByBoundingSphere(
        (ref_coords[0] + LatticeVec[0][0], ref_coords[1], ref_coords[2] + LatticeVec[2][2]), box_tol)
    vertex_g = m.rootAssembly.sets[NameSet].nodes.getByBoundingSphere(
        (ref_coords[0] + LatticeVec[0][0], ref_coords[1] + LatticeVec[1][1], ref_coords[2] + LatticeVec[2][2]),
        box_tol)
    vertex_h = m.rootAssembly.sets[NameSet].nodes.getByBoundingSphere(
        (ref_coords[0], ref_coords[1] + LatticeVec[1][1], ref_coords[2] + LatticeVec[2][2]), box_tol)

    mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-vertex-d', nodeLabels=(
    (vertex_d[0].instanceName, (vertex_d[0].label,)),))
    mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-vertex-a', nodeLabels=(
    (vertex_a[0].instanceName, (vertex_a[0].label,)),))
    for Dim1 in [1, 2, 3]:
        mdb.models[NameModel].Equation(name='PBC-d-a' + str(Dim1),
                                        terms=((1.0, 'PBC-vertex-d', Dim1), (-1.0, 'PBC-vertex-a', Dim1),
                                                (-1.0, RP_Names[1], Dim1)))
    mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-vertex-h', nodeLabels=(
    (vertex_h[0].instanceName, (vertex_h[0].label,)),))
    mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-vertex-e', nodeLabels=(
    (vertex_e[0].instanceName, (vertex_e[0].label,)),))
    for Dim1 in [1, 2, 3]:
        mdb.models[NameModel].Equation(name='PBC-h-e' + str(Dim1),
                                        terms=((1.0, 'PBC-vertex-h', Dim1), (-1.0, 'PBC-vertex-e', Dim1),
                                                (-1.0, RP_Names[1], Dim1)))
    mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-vertex-g', nodeLabels=(
    (vertex_g[0].instanceName, (vertex_g[0].label,)),))
    mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-vertex-f', nodeLabels=(
    (vertex_f[0].instanceName, (vertex_f[0].label,)),))
    for Dim1 in [1, 2, 3]:
        mdb.models[NameModel].Equation(name='PBC-g-f' + str(Dim1),
                                        terms=((1.0, 'PBC-vertex-g', Dim1), (-1.0, 'PBC-vertex-f', Dim1),
                                                (-1.0, RP_Names[1], Dim1)))
    mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-vertex-c', nodeLabels=(
    (vertex_c[0].instanceName, (vertex_c[0].label,)),))
    mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-vertex-b', nodeLabels=(
    (vertex_b[0].instanceName, (vertex_b[0].label,)),))
    for Dim1 in [1, 2, 3]:
        mdb.models[NameModel].Equation(name='PBC-c-b' + str(Dim1),
                                        terms=((1.0, 'PBC-vertex-c', Dim1), (-1.0, 'PBC-vertex-b', Dim1),
                                                (-1.0, RP_Names[1], Dim1)))
    mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-vertex-b', nodeLabels=(
    (vertex_b[0].instanceName, (vertex_b[0].label,)),))
    mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-vertex-a', nodeLabels=(
    (vertex_a[0].instanceName, (vertex_a[0].label,)),))
    for Dim1 in [1, 2, 3]:
        mdb.models[NameModel].Equation(name='PBC-b-a' + str(Dim1),
                                        terms=((1.0, 'PBC-vertex-b', Dim1), (-1.0, 'PBC-vertex-a', Dim1),
                                                (-1.0, RP_Names[0], Dim1)))
    mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-vertex-f', nodeLabels=(
    (vertex_f[0].instanceName, (vertex_f[0].label,)),))
    mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-vertex-e', nodeLabels=(
    (vertex_e[0].instanceName, (vertex_e[0].label,)),))
    for Dim1 in [1, 2, 3]:
        mdb.models[NameModel].Equation(name='PBC-f-e' + str(Dim1),
                                        terms=((1.0, 'PBC-vertex-f', Dim1), (-1.0, 'PBC-vertex-e', Dim1),
                                                (-1.0, RP_Names[0], Dim1)))
    mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-vertex-e', nodeLabels=(
    (vertex_e[0].instanceName, (vertex_e[0].label,)),))
    mdb.models[NameModel].rootAssembly.SetFromNodeLabels(name='PBC-vertex-a', nodeLabels=(
    (vertex_a[0].instanceName, (vertex_a[0].label,)),))
    for Dim1 in [1, 2, 3]:
        mdb.models[NameModel].Equation(name='PBC-e-a' + str(Dim1),
                                        terms=((1.0, 'PBC-vertex-e', Dim1), (-1.0, 'PBC-vertex-a', Dim1),
                                                (-1.0, RP_Names[2], Dim1)))

    print("Time to create PBC: " + str(end3 - start3))
    return (
    mdb.models[NameModel].rootAssembly.sets[NameSet].nodes.getFromLabel(x_pos_coord[0][3]).coordinates, NameRef1,
    NameRef2, NameRef3)


######### Ende Code von Noah

def define_node_sets(m, max_radius, lateral_cube_size):
    #Definieren des NodeSets, in dem Modell kann optisch ueberprueft werden, ob die Nodes richtig ausgewaehlt werden, dazu muss man das Node Set "NSET_Mat_Surface" unter Assembly im Mesh anzeigen lassen
    mat_part = m.parts['CombinedGeometry']
    n = mat_part.nodes
    nodes = n.getByBoundingBox(zMin=-max_radius-0.001, yMin=-max_radius-0.001, xMin=-max_radius-0.001,
                               zMax=-max_radius+0.001, yMax=thickness, xMax=lateral_cube_size) + \
    n.getByBoundingBox(zMin=-max_radius-0.001+lateral_cube_size, yMin=-max_radius-0.001, xMin=-max_radius-0.001,
                       zMax=-max_radius+0.001+lateral_cube_size, yMax=thickness, xMax=lateral_cube_size) + \
    n.getByBoundingBox(zMin=-max_radius-0.001, yMin=-max_radius-0.001, xMin=-max_radius-0.001,
                       zMax=lateral_cube_size, yMax=lateral_cube_size, xMax=-max_radius+0.001) + \
    n.getByBoundingBox(zMin=-max_radius-0.001, yMin=-max_radius-0.001, xMin=-max_radius-0.001+lateral_cube_size,
                       zMax=lateral_cube_size, yMax=lateral_cube_size, xMax=-max_radius+0.001+lateral_cube_size)

    mat_part.Set(nodes=nodes, name='NSET_Mat_Surface')
    
def finalize_abaqus_job(m, setup_data, simulation_counter):
        ###### Im Folgenden Abschnitt werden ausgaben angefragt, der Job erstellt und die Leistungs reservierung definiert ########

    #Anfragen der gewuenschten Ergebnisse
    m.FieldOutputRequest(
        name='DiffusionOutput',
        createStepName='DiffusionStep',
        variables=('CONC', 'NT')  #Konzentration und den Fluss
    )

    # Job erstellen und mit 31 CPU-Kernen und 90% Speicherreservierung konfigurieren
    job_name = setup_data['job_name']
    mdb.Job(name=job_name, model= setup_data['ModelName'], description=f"Simulation {setup_data['simulation_id']}")
    mdb.jobs[job_name].setValues(numCpus=31, numDomains=31, multiprocessingMode=DEFAULT)
    mdb.jobs[job_name].setValues(memory=90, memoryUnits=PERCENTAGE)

    #Job einreichen und warten, bis er abgeschlossen ist
    mdb.jobs[job_name].submit(consistencyChecking=OFF)
    mdb.jobs[job_name].waitForCompletion()

    #Post-process: oefnnen und anzeigen der Ergebnisse
    odb_path = f'{job_name}.odb'
    session.openOdb(name=odb_path)
    session.viewports['Viewport: 1'].setValues(displayedObject=session.odbs[odb_path])
    session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF,))

    #Setze die Frame-Rate auf "Slow" (niedrige Geschwindigkeit)
    session.animationOptions.setValues(frameRate=6)

    # Zugriff auf den aktuellen Viewport
    viewport = session.viewports['Viewport: 1']

    '''
    #Drehe die Ansicht / aktuell nicht notwendig
    viewport.view.rotate(
        xAngle=0.0,  # Drehe 90 Grad um die X-Achse
        yAngle=0.0,  # Keine Drehung um die Y-Achse
        zAngle=0.0,  # Keine Drehung um die Z-Achse
        mode=MODEL,  # Rotationsmodus relativ zum Modell
        drawImmediately=True  # Aktualisiere die Ansicht sofort
    )
    '''

    #Ansicht anpassen, damit das gesamte Modell sichtbar ist
    viewport.view.fitView()

    #Setze den Animationstyp auf SCALE_FACTOR
    session.viewports['Viewport: 1'].animationController.setValues(animationType=SCALE_FACTOR)

    #Starte die Animation
    session.viewports['Viewport: 1'].animationController.play()

    ##### Der Folgende Abschnitt befasst sich mit dem Speichern der Ergebniss Dateien ######

    # ODB-Datei kopieren und im neuen Ordner speichern
    odb_file = setup_data['odb_file']
    current_path = os.path.join(r"C:\temp", odb_file)
    destination_path = os.path.join(results_folder, odb_file)
    try:
        if not os.path.exists(current_path):
            print(f"Fehler: ODB-Datei {odb_file} nicht gefunden.")
        else:
            shutil.copy2(current_path, destination_path)
            print(f"ODB-File copied successfully: {destination_path}")

        '''
        # Loesche das Modell nach der Speicherung der ODB, aktuell ausgeklammert sollte am Ende wieder eingeschaltet werden, um Abaqus zu entlasten
        if ModelName in mdb.models:
            del mdb.models[ModelName]
            print(f"Model {ModelName} was deleted.")
        '''

    except Exception as e:
        print(f"Fehler beim Kopieren der ODB-Datei: {e}")

    end_time = time.time()
    total_time = end_time - setup_data['start_time']
    print(f"Simulation {simulation_counter} was running {total_time:.2f} seconds.")


if __name__ == "__main__":
    cube_material_name = 'CubeMaterial'
    cube_section_name = 'CubeSection'

    thickness = 5
    sub_path = f'{thickness} nm'

    # Ordner mit CSV-Dateien
    csv_folder = f'C:/Users/franke/source/repos/JCFranke2023/RVE/raw_data/PlasmaTech25/{sub_path}/Pore List/'

    # Durch alle CSV-Dateien iterieren
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

    # Erstellt Ordner fuer Ergebnisse
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_folder = os.path.join(f'C:/Users/franke/source/repos/JCFranke2023/RVE/results/PlasmaTech25/{sub_path}/', f"{current_datetime}")
    os.makedirs(results_folder, exist_ok=True)

    for simulation_counter, csv_file in enumerate(csv_files):
        setup_data = setup_abaqus_environment(csv_folder, csv_file)
        m = setup_data['m']
        Inclusions, lateral_cube_size, max_radius = define_cube_dimensions(setup_data)
        create_cube_part(m, lateral_cube_size, max_radius)
        instances_list = init_pores(m, Inclusions)
        combined_part = merge_cube_and_pores(m, lateral_cube_size)
        assembly = setup_boundary_conditions(m, lateral_cube_size)
        generate_mesh(combined_part)
        define_node_sets(m, max_radius, lateral_cube_size)
        PeriodicBound3D(mdb, setup_data['ModelName'], "NSET_Mat_Surface", lateral_cube_size)
        finalize_abaqus_job(m, setup_data, simulation_counter)






    ##### In diesem Abschnitt werden die Oberflaechen des Wuerfels Partitioniert, um ein homogenes Mesh auf der Oberflaeche zu gewaehrleisten #####
    # hat nicht funktioniert laut Clemens
    '''
    # Definition der Partitionsebenen
    # Zugriff auf das Modell
    model = mdb.models[ModelName]
    assembly = model.rootAssembly
    instance = assembly.instances['CombinedGeometry-1']

    # Definition der Partitionsebenen
    for face in [
        (cube_width / 2, -max_radius, cube_length / 2),
        (cube_width / 2, cube_height - max_radius, cube_length / 2),
        (cube_width / 2, cube_height / 2, -max_radius),
        (cube_width / 2, cube_height / 2, cube_length - max_radius)
    ]:

        # Finden der zu partitionierenden Flaeche
        target_face = assembly.instances['CombinedGeometry-1'].faces.getClosest(coordinates = ((0.0, 0.0, 0.0),))

        if not target_face:
            raise ValueError(f"Fehler: Flaeche mit Koordinaten {face} nicht gefunden.")
        
        x = target_face[0][0].index
        print(assembly.instances['CombinedGeometry-1'])
        print(type(assembly.instances['CombinedGeometry-1']))
        d = assembly.instances['CombinedGeometry-1'].datums[2]

        assembly.instances['CombinedGeometry-1'].PartitionFaceByDatumPlane(faces = p.faces[x:x+1], datumPlane = d)



    print("Cube succesfull partioned.")
    '''


    # was macht das hier?
    '''
    # Zugriff auf die Assembly und das vermaschte Part
    #assembly = m.rootAssembly
    #combined_part = m.parts['CombinedGeometry']
    #combined_instance = assembly.instances['CombinedGeometry-1']  # Vermaschte Instanz

    # Alle Flaechen des vermaschten Teils abrufen
    all_faces = combined_part.faces  # Holt alle Flaechen aus dem Mesh

    # Setze eine Toleranz fuer die Flaechenkoordinaten
    tolerance = 1e-3

    # Initialisiere die Variablen fuer die angrenzenden Flaechen
    face_1 = None
    face_2 = None

    # Durchlaufe alle Flaechen und suche zwei angrenzende
    for face in all_faces:
        for other_face in all_faces:
            if face == other_face:
                continue  # Skip, wenn es die gleiche Flaeche ist

            # Extrahiere die Knotennummern (Labels) der Flaechen
            nodes_face_1 = [node.label for node in face.getNodes()]
            nodes_face_2 = [node.label for node in other_face.getNodes()]

            # Pruefe, ob zwei Flaechen gemeinsame Knotennummern haben
            common_nodes = set(nodes_face_1) & set(nodes_face_2)

            if len(common_nodes) > 0:  # Falls es gemeinsame Knoten gibt
                face_1 = face
                face_2 = other_face
                break  # Stoppe die Schleife, sobald ein Paar gefunden wurde

        if face_1 and face_2:
            break  # Stoppe auch die aeussere Schleife

    ###### BLACK BOX Ende #######
    '''
    ###### Im folgenden Abschnitt werden die Constraints erstellt ##########



    #Funktionsaufruf

print(f"All {simulation_counter + 1} Simulations finished successfully")
