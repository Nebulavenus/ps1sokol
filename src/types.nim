import sokol/gfx as sg
import math/vec3
import math/mat4
import tables

type
  Vertex* = object
    x*, y*, z*: float32
    xN*, yN*, zN*: float32
    color*: uint32
    u*, v*: float32
    bxN*, byN*, bzN*: float32 # Bent Normal vector

  SpriteVertex* = object
    x*, y*, z*: float32
    color*: uint32
    u*, v*: float32

  Mesh* = object
    bindings*: Bindings
    indexCount*: int32
    bounds*: AABB

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

  Difficulty* = enum
    Easy,
    Medium,
    Hard

  AIVehicle* = object
    position*: Vec3
    velocity*: Vec3
    rotation*: Mat4
    yaw*: float32
    angularVelocity*: float32
    targetNode*: int
    currentCheckpointIdx*: int
    lapCount*: int
    difficulty*: Difficulty
    name*: string

  Checkpoint* = object
    pos*: Vec3
    radius*: float32

  ReplayFrame* = object
    pos*: Vec3
    yaw*: float32

  InputState* = object
    accelerate*: bool
    brake*: bool
    turnLeft*: bool
    turnRight*: bool
    drift*: bool

  ResourceManager* = object
    meshes*: Table[string, Mesh]
    images*: Table[string, sg.Image]
    samplers*: Table[string, sg.Sampler]

  CameraMode* = enum
    Follow,
    Front

  GameState* = enum
    MainMenu,
    Playing,
    Paused,
    ControlsMenu

  PauseMenuState* = object
    selectedItem*: int
    itemCount*: int

  State* = object
    # App
    gameHasFocus*: bool
    gameState*: GameState
    previousGameState*: GameState # To return from sub-menus
    menu*: PauseMenuState
    # Asset Management
    res*: ResourceManager
    # Rendering
    pip*: Pipeline
    pipSprite*: Pipeline
    pipPost*: Pipeline
    offscreenImg*: sg.Image
    offscreenDepthImg*: sg.Image
    offscreenAttachments*: sg.Attachments
    offscreenSampler*: sg.Sampler
    offscreenPassAction*: sg.PassAction
    screenVBuf*: sg.Buffer
    screenIBuf*: sg.Buffer
    shadowTexture*: sg.Image
    shadowSampler*: sg.Sampler
    particleTexture*: sg.Image
    checkpointTexture*: sg.Image
    quadVBuf*: sg.Buffer
    quadIBuf*: sg.Buffer
    passAction*: sg.PassAction
    # Track meshes
    trackMesh1*: Mesh # Road
    trackMesh2*: Mesh # Shape
    trackMesh3*: Mesh # Grass
    trackMesh4*: Mesh # Trees
    # CPU-side collision geometry for the road
    roadCollisionVertices*: seq[Vertex]
    roadCollisionIndices*: seq[uint16]
    roadGrid*: UniformGrid
    barrierCollisionVertices*: seq[Vertex]
    barrierCollisionIndices*: seq[uint16]
    barrierGrid*: UniformGrid
    # Player's car mesh
    carMesh1*: Mesh # Body
    carMesh2*: Mesh # Rear wheel
    carMesh3*: Mesh # Front wheel
    # Input & Logic
    input*: InputState
    player*: PlayerVehicle
    cameraMode*: CameraMode
    cameraOffsetY*: float32
    cameraPos*: Vec3 # Camera's actual world position
    cameraTarget*: Vec3 # Point the camera is looking at
    particles*: ParticleSystem
    lastEmitPos*: Vec3
    time*: float32
    # Gameplay
    checkpoints*: seq[Checkpoint]
    currentCheckpointIdx*: int
    lapCount*: int
    lapStartTime*: float32
    bestLapTime*: float32
    lastLapTime*: float32
    # AI
    aiCars*: seq[AIVehicle]
    pathNodes*: seq[Vec3]
    # Replay
    replayBuffer*: seq[ReplayFrame]
    isReplaying*: bool
    replayIndex*: int
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

  Particle* = object
    pos*: Vec3
    vel*: Vec3
    rot*: float32
    rotVel*: float32
    color*: uint32
    life*: float32
    maxLife*: float32

  ParticleSystem* = object
    pool*: array[1024, Particle]
    nextIndex*: int
