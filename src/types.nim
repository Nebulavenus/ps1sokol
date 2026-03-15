import sokol/gfx as sg
import math/vec3
import math/mat4

type
  Vertex* = object
    x*, y*, z*: float32
    xN*, yN*, zN*: float32
    color*: uint32
    u*, v*: float32
    bxN*, byN*, bzN*: float32 # Bent Normal vector

  Mesh* = object
    bindings*: Bindings
    indexCount*: int32

  # Axis-Aligned Bounding Box
  AABB* = object
    min*, max*: Vec3

  # Uniform Grid acceleration structure
  UniformGrid* = object
    bounds*: AABB
    dims*: (int, int, int) # Number of cells in each dimension (x, y, z)
    cellSize*: Vec3        # Size of a single cell in world units
    cells*: seq[seq[int]]  # 1D list of cells, each cell is a list of triangle

  AOBakeParams* = object
    # Number of rays to cast per vertex. More is better but slower. (e.g., 64, 128)
    numRays*: int
    # How far a ray can travel to cause occlusion. Prevents distant geometry from affecting local AO.
    maxDistance*: float
    # How strong the darkening effect is. (e.g., 1.0)
    intensity*: float
    # A small offset to push the ray origin away from the vertex to prevent self-intersection. (e.g., 0.001)
    bias*: float

  PlayerVehicle* = object
    position*: Vec3
    velocity*: Vec3
    rotation*: Mat4
    yaw*: float32
    angularVelocity*: float32

  InputState* = object
    accelerate*: bool
    brake*: bool
    turnLeft*: bool
    turnRight*: bool
    drift*: bool

  State* = object
    # App
    gameHasFocus*: bool
    # Rendering
    pip*: Pipeline
    passAction*: sg.PassAction
    # Track meshes
    trackMesh1*: Mesh # Road
    trackMesh2*: Mesh # Shape
    trackMesh3*: Mesh # Grass
    trackMesh4*: Mesh # Trees
    # CPU-side collision geometry for the road
    roadCollisionVertices*: seq[Vertex]
    roadCollisionIndices*: seq[uint16]
    barrierCollisionVertices*: seq[Vertex]
    barrierCollisionIndices*: seq[uint16]
    # Player's car mesh
    carMesh1*: Mesh # Body
    carMesh2*: Mesh # Rear wheel
    carMesh3*: Mesh # Front wheel
    # Input & Logic
    input*: InputState
    player*: PlayerVehicle
    cameraOffsetY*: float32
    cameraPos*: Vec3 # Camera's actual world position
    cameraTarget*: Vec3 # Point the camera is looking at
    # From audio.nim calculated values
    debugSpeed*: float32
    debugRpm*: float32
    debugGear*: int32
    # -- Controlling AO Multi-Layered --
    aoShadowStrength*: float32
    skyLightColor*: Vec3
    skyLightIntensity*: float32
    groundLightColor*: Vec3
    groundLightIntensity*: float32

  SurfaceHit* = object
    pos*: Vec3
    normal*: Vec3

  CollisionResponse* = object
    pushOut*: Vec3
    collided*: bool
