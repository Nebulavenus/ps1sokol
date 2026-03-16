import sokol/log as slog
import sokol/app as sapp
import sokol/gfx as sg
import sokol/glue as sglue
import sokol/debugtext as sdtx
import shaders/default as shd
import shaders/sprite as spr
import shaders/postfx as pfx
import math
import math/vec2
import math/vec3
import math/mat4
import strutils
import os
import std/random
import std/strformat
import std/algorithm
import audio
import rtfs
import colors
import types
import mesh_loader
import aobaker
import physics
import options
import culling

when defined(emscripten):
  proc emscripten_run_script(script: cstring) {.importc, header: "<emscripten/emscripten.h>".}

var ASSETS_FS: RuntimeFS
var state: State

const
  passAction = PassAction(
    colors: [
      ColorAttachmentAction(
        loadAction: loadActionClear,
        clearValue: (0.25, 0.5, 0.75, 1)
      )
    ]
  )

proc extractPathFromRoadMesh(vertices: seq[Vertex], indices: seq[uint16]): seq[Vec3] =
  var rawNodes: seq[Vec3]
  for i in countup(0, indices.len - 3, 3):
    let v0 = vertices[indices[i+0]]
    let v1 = vertices[indices[i+1]]
    let v2 = vertices[indices[i+2]]
    let center = vec3((v0.x + v1.x + v2.x) / 3.0, (v0.y + v1.y + v2.y) / 3.0, (v0.z + v1.z + v2.z) / 3.0)
    rawNodes.add(center)
  
  if rawNodes.len == 0: return @[]

  var sortedNodes: seq[Vec3]
  var visited = newSeq[bool](rawNodes.len)
  
  # 1. Start by finding the ABSOLUTE closest triangle center to the spawn position
  var currentPos = vec3(0.0, 12.0, 25.0)
  var startIdx = -1
  var startMinDist = 1e9
  for i in 0 ..< rawNodes.len:
    let d = len(rawNodes[i] - currentPos)
    if d < startMinDist:
      startMinDist = d
      startIdx = i
  
  if startIdx != -1:
    currentPos = rawNodes[startIdx]
    # Do NOT mark visited yet, so the first spine node includes it
  
  # Heuristic: Find the spine of the road by jumping along it and averaging nearby triangles
  # Increased limit to 500 nodes to support longer tracks
  for _ in 0 ..< 500:
    var closestIdx = -1
    var closestDist = 1e9
    
    # Find the nearest unvisited triangle that is at least a small jump away (5.0)
    # to avoid clustering, but not too far (35.0) to avoid jumping to parallel roads
    for i in 0 ..< rawNodes.len:
      if visited[i]: continue
      let d = len(rawNodes[i] - currentPos)
      if d < closestDist and d > 5.0: # Minimum jump reduced for better resolution
        closestDist = d
        closestIdx = i
    
    if closestIdx != -1:
      # Now, find ALL triangles within a "road width" of this closest point
      # and average them to find the true center (the spine)
      var sum = vec3(0,0,0)
      var count = 0
      let anchor = rawNodes[closestIdx]
      for i in 0 ..< rawNodes.len:
        if visited[i]: continue
        let d = len(rawNodes[i] - anchor)
        if d < 25.0: # Road width threshold slightly increased
          sum = sum + rawNodes[i]
          count += 1
          visited[i] = true
      
      if count > 0:
        let spineNode = sum / count.float
        sortedNodes.add(spineNode)
        currentPos = spineNode
    else:
      break

  # Path Smoothing: Interpolate between nodes using Catmull-Rom to create a "curve"
  if sortedNodes.len > 4:
    var curvedNodes: seq[Vec3]
    for i in 0 ..< sortedNodes.len:
      let p0 = sortedNodes[(i + sortedNodes.len - 1) mod sortedNodes.len]
      let p1 = sortedNodes[i]
      let p2 = sortedNodes[(i + 1) mod sortedNodes.len]
      let p3 = sortedNodes[(i + 2) mod sortedNodes.len]
      
      # Subdivide each segment into 4 parts for smoother AI pathing
      for t_idx in 0 ..< 4:
        let t = t_idx.float32 / 4.0f32
        let tt = t * t
        let ttt = tt * t
        
        let node = (
          (p1 * 2.0f32) +
          (p2 - p0) * t +
          (p0 * 2.0f32 - p1 * 5.0f32 + p2 * 4.0f32 - p3) * tt +
          (p1 * 3.0f32 - p0 - p2 * 3.0f32 + p3) * ttt
        ) * 0.5f32
        curvedNodes.add(node)
    
    echo &"Extracted {sortedNodes.len} spine nodes, expanded to {curvedNodes.len} curved nodes"
    return curvedNodes

  echo &"Extracted {sortedNodes.len} sequential center-line nodes"
  return sortedNodes

proc computeFsParams(): shd.FsParams =
  result = shd.FsParams(
    u_fogColor: vec3(0.25f, 0.5f, 0.75f),
    u_fogNear: 4.0f,
    u_fogFar: 150.0f,
    u_ditherSize: vec2(sapp.widthf(), sapp.heightf()),
    u_aoShadowStrength: state.aoShadowStrength,
    u_skyLightColor: state.skyLightColor,
    u_skyLightIntensity: state.skyLightIntensity,
    u_groundLightColor: state.groundLightColor,
    u_groundLightIntensity: state.groundLightIntensity
  )

proc makeCircularTexture(size: int, alphaMultiplier: float32 = 255.0): sg.Image =
  var pixels = newSeq[uint32](size * size)
  for y in 0 ..< size:
    for x in 0 ..< size:
      let dx = x.float - size.float * 0.5
      let dy = y.float - size.float * 0.5
      let dist = sqrt(dx*dx + dy*dy)
      
      var alpha = 0.0
      alpha += clamp(1.0 - (dist / (size.float * 0.4)), 0.0, 1.0)
      
      let a8 = (clamp(alpha, 0.0, 1.0) * alphaMultiplier).uint8
      pixels[y * size + x] = packColor(255, 255, 255, a8)

  result = sg.makeImage(sg.ImageDesc(
    width: size.int32,
    height: size.int32,
    pixelFormat: pixelFormatRgba8,
    data: ImageData(
      subimage: [ [ sg.Range(addr: pixels[0].addr, size: pixels.len * 4) ] ]
    )
  ))

proc initParticleTexture() =
  state.particleTexture = makeCircularTexture(32, 255.0)
  state.checkpointTexture = makeCircularTexture(64, 100.0)

proc emitParticle(pos, vel: Vec3, color: uint32, life: float32) =
  let idx = state.particles.nextIndex
  state.particles.pool[idx] = Particle(
    pos: pos,
    vel: vel,
    color: color,
    life: life,
    maxLife: life
  )
  state.particles.nextIndex = (idx + 1) mod state.particles.pool.len

proc updateParticles(dt: float32) =
  for i in 0 ..< state.particles.pool.len:
    var p = addr state.particles.pool[i]
    if p.life > 0:
      p.life -= dt
      p.pos = p.pos + (p.vel * dt)
      p.vel = p.vel * 0.95 # Air resistance

proc drawParticles(proj, view: Mat4) =
  sg.applyPipeline(state.pipSprite)
  
  var bindings = Bindings()
  bindings.vertexBuffers[0] = state.quadVBuf
  bindings.indexBuffer = state.quadIBuf
  bindings.images[spr.imgUTexture] = state.particleTexture
  bindings.samplers[spr.smpUSampler] = state.shadowSampler
  sg.applyBindings(bindings)

  # Distance Sorting
  type ParticleDist = object
    idx: int
    distSq: float32

  var activeParticles: seq[ParticleDist]
  for i in 0 ..< state.particles.pool.len:
    if state.particles.pool[i].life > 0:
      let d2 = lenSqr(state.particles.pool[i].pos - state.cameraPos)
      activeParticles.add(ParticleDist(idx: i, distSq: d2))
  
  # Sort furthest to nearest
  activeParticles.sort(proc (a, b: ParticleDist): int =
    if a.distSq > b.distSq: -1
    elif a.distSq < b.distSq: 1
    else: 0
  )

  for pd in activeParticles:
    let p = state.particles.pool[pd.idx]
    
    let camForward = norm(state.cameraTarget - state.cameraPos)
    let camRight = norm(cross(camForward, vec3.up()))
    let camUp = norm(cross(camRight, camForward))
    
    let s = (p.life / p.maxLife) * 0.8
    let particleModel = fromCols(camRight * s, camUp * s, camForward * s, p.pos)
    
    var vsParams = spr.VsParams(
      u_mvp: proj * view * particleModel,
      u_camPos: state.cameraPos,
      u_jitterAmount: 240.0
    )
    
    sg.applyUniforms(spr.ubVsParams, sg.Range(addr: vsParams.addr, size: vsParams.sizeof))
    
    var fsParams = spr.FsParams(
      u_fogColor: vec3(0.25f, 0.5f, 0.75f),
      u_fogNear: 50.0f, # Push fog back for particles
      u_fogFar: 150.0f,
      u_alphaThreshold: 0.1f
    )
    sg.applyUniforms(spr.ubFsParams, sg.Range(addr: fsParams.addr, size: fsParams.sizeof))
    sg.draw(0, 6, 1)

proc initShadowTexture() =
  const size = 64
  var pixels = newSeq[uint32](size * size)
  for y in 0 ..< size:
    for x in 0 ..< size:
      let dx = x.float - size.float * 0.5
      let dy = y.float - size.float * 0.5
      let dist = sqrt(dx*dx + dy*dy)
      let radius = size.float * 0.4
      let alpha = clamp(1.0 - (dist / radius), 0.0, 1.0)
      let a8 = (alpha * 150.0).uint8 # Semi-transparent black
      pixels[y * size + x] = packColor(0, 0, 0, a8)

  state.shadowTexture = sg.makeImage(sg.ImageDesc(
    width: size,
    height: size,
    pixelFormat: pixelFormatRgba8,
    data: ImageData(
      subimage: [ [ sg.Range(addr: pixels[0].addr, size: pixels.len * 4) ] ]
    )
  ))
  state.shadowSampler = sg.makeSampler(sg.SamplerDesc(
    minFilter: filterLinear,
    magFilter: filterLinear,
  ))

proc initQuadBuffers() =
  let vertices = [
    SpriteVertex(x: -1.0, y: 0.0, z: -1.0, color: 0xFFFFFFFF'u32, u: 0.0, v: 0.0),
    SpriteVertex(x:  1.0, y: 0.0, z: -1.0, color: 0xFFFFFFFF'u32, u: 1.0, v: 0.0),
    SpriteVertex(x:  1.0, y: 0.0, z:  1.0, color: 0xFFFFFFFF'u32, u: 1.0, v: 1.0),
    SpriteVertex(x: -1.0, y: 0.0, z:  1.0, color: 0xFFFFFFFF'u32, u: 0.0, v: 1.0),
  ]
  state.quadVBuf = sg.makeBuffer(BufferDesc(
    usage: BufferUsage(vertexBuffer: true),
    data: sg.Range(addr: vertices[0].addr, size: vertices.sizeof)
  ))
  
  let indices: array[6, uint16] = [ 0, 1, 2, 0, 2, 3 ]
  state.quadIBuf = sg.makeBuffer(BufferDesc(
    usage: BufferUsage(indexBuffer: true),
    data: sg.Range(addr: indices[0].addr, size: indices.sizeof)
  ))

proc initSpritePipeline() =
  state.pipSprite = sg.makePipeline(PipelineDesc(
    shader: sg.makeShader(spr.spriteShaderDesc(sg.queryBackend())),
    layout: VertexLayoutState(
      attrs: [
        VertexAttrState(format: vertexFormatFloat3), # a_position
        VertexAttrState(format: vertexFormatUbyte4n),# a_color0
        VertexAttrState(format: vertexFormatFloat2), # a_texcoord0
      ],
    ),
    indexType: indexTypeUint16,
    cullMode: cullModeNone,
    depth: DepthState(
      compare: compareFuncLessEqual,
      writeEnabled: false, # Shadows don't write to depth
    ),
    colors: [
      ColorTargetState(
        blend: BlendState(
          enabled: true,
          srcFactorRgb: blendFactorSrcAlpha,
          dstFactorRgb: blendFactorOneMinusSrcAlpha,
          srcFactorAlpha: blendFactorOne,
          dstFactorAlpha: blendFactorZero
        )
      )
    ]
  ))

proc initOffscreen() =
  state.offscreenImg = sg.makeImage(sg.ImageDesc(
    usage: ImageUsage(renderAttachment: true),
    width: 640,
    height: 480,
    pixelFormat: pixelFormatRgba8,
    sampleCount: 1
  ))
  
  state.offscreenDepthImg = sg.makeImage(sg.ImageDesc(
    usage: ImageUsage(renderAttachment: true),
    width: 640,
    height: 480,
    pixelFormat: pixelFormatDepth,
    sampleCount: 1
  ))

  state.offscreenSampler = sg.makeSampler(sg.SamplerDesc(
    minFilter: filterNearest,
    magFilter: filterNearest,
  ))

  var attDesc = AttachmentsDesc()
  attDesc.colors[0].image = state.offscreenImg
  attDesc.depthStencil.image = state.offscreenDepthImg
  state.offscreenAttachments = sg.makeAttachments(attDesc)
  
  state.offscreenPassAction = PassAction(
    colors: [
      ColorAttachmentAction(
        loadAction: loadActionClear,
        clearValue: (0.25, 0.5, 0.75, 1)
      )
    ],
    depth: DepthAttachmentAction(
      loadAction: loadActionClear,
      clearValue: 1.0
    )
  )

proc initScreenQuad() =
  let vertices = [
    # x, y, u, v
    -1.0f, -1.0f,  0.0f, 1.0f,
     1.0f, -1.0f,  1.0f, 1.0f,
     1.0f,  1.0f,  1.0f, 0.0f,
    -1.0f,  1.0f,  0.0f, 0.0f,
  ]
  state.screenVBuf = sg.makeBuffer(BufferDesc(
    usage: BufferUsage(vertexBuffer: true),
    data: sg.Range(addr: vertices[0].addr, size: vertices.sizeof)
  ))
  
  let indices: array[6, uint16] = [ 0, 1, 2, 0, 2, 3 ]
  state.screenIBuf = sg.makeBuffer(BufferDesc(
    usage: BufferUsage(indexBuffer: true),
    data: sg.Range(addr: indices[0].addr, size: indices.sizeof)
  ))

proc initPostfxPipeline() =
  state.pipPost = sg.makePipeline(PipelineDesc(
    shader: sg.makeShader(pfx.postfxShaderDesc(sg.queryBackend())),
    layout: VertexLayoutState(
      attrs: [
        VertexAttrState(format: vertexFormatFloat2), # position
        VertexAttrState(format: vertexFormatFloat2), # texcoord0
      ],
    ),
    indexType: indexTypeUint16,
    colors: [ ColorTargetState() ]
  ))

proc drawPostfx() =
  var bindings = Bindings()
  bindings.vertexBuffers[0] = state.screenVBuf
  bindings.indexBuffer = state.screenIBuf
  bindings.images[pfx.imgUTexture] = state.offscreenImg
  bindings.samplers[pfx.smpUSampler] = state.offscreenSampler
  
  var fsParams = pfx.FsParams(
    u_resolution: [640.0f, 480.0f],
    u_time: state.time
  )
  
  sg.applyPipeline(state.pipPost)
  sg.applyBindings(bindings)
  sg.applyUniforms(pfx.ubFsParams, sg.Range(addr: fsParams.addr, size: fsParams.sizeof))
  sg.draw(0, 6, 1)

proc drawCheckpoints(proj, view: Mat4) =
  if state.checkpoints.len == 0: return
  
  # Only draw the next checkpoint to guide the player
  let cp = state.checkpoints[state.currentCheckpointIdx]
  
  # Draw a vertical "gate" or "beam"
  let camForward = norm(state.cameraTarget - state.cameraPos)
  let camRight = norm(cross(camForward, vec3.up()))
  
  # Scale it to be a tall pillar
  let cpModel = translate(cp.pos + vec3(0, 5, 0)) * fromCols(camRight * cp.radius, vec3.up() * 10.0, camForward * cp.radius, vec3(0,0,0))
  
  var vsParams = spr.VsParams(
    u_mvp: proj * view * cpModel,
    u_camPos: state.cameraPos,
    u_jitterAmount: 240.0
  )
  
  var fsParams = spr.FsParams(
    u_fogColor: vec3(0.25f, 0.5f, 0.75f),
    u_fogNear: 50.0f,
    u_fogFar: 150.0f,
    u_alphaThreshold: 0.01f
  )
  
  var bindings = Bindings()
  bindings.vertexBuffers[0] = state.quadVBuf
  bindings.indexBuffer = state.quadIBuf
  bindings.images[spr.imgUTexture] = state.checkpointTexture
  bindings.samplers[spr.smpUSampler] = state.shadowSampler
  
  sg.applyPipeline(state.pipSprite)
  sg.applyBindings(bindings)
  sg.applyUniforms(spr.ubVsParams, sg.Range(addr: vsParams.addr, size: vsParams.sizeof))
  sg.applyUniforms(spr.ubFsParams, sg.Range(addr: fsParams.addr, size: fsParams.sizeof))
  sg.draw(0, 6, 1)

proc drawShadow(proj, view: Mat4) =
  let surfaceHitOpt = getSurfaceInfo(state, state.player.position)
  if surfaceHitOpt.isNone: return
  let hit = surfaceHitOpt.get()
  let shadowPos = hit.pos + hit.normal * 0.05
  let carRotationOnlyYaw = rotate(state.player.yaw, vec3(0, 1, 0))
  let shadowModel = translate(shadowPos) * carRotationOnlyYaw * scale(vec3(1.5, 1.0, 2.5))
  
  var vsParams = spr.VsParams(
    u_mvp: proj * view * shadowModel,
    u_camPos: state.cameraPos,
    u_jitterAmount: 240.0
  )
  var fsParams = spr.FsParams(
    u_fogColor: vec3(0.25f, 0.5f, 0.75f),
    u_fogNear: 1000.0f, # Disable fog for shadow
    u_fogFar: 1500.0f,
    u_alphaThreshold: 0.01f
  )
  var bindings = Bindings()
  bindings.vertexBuffers[0] = state.quadVBuf
  bindings.indexBuffer = state.quadIBuf
  bindings.images[spr.imgUTexture] = state.shadowTexture
  bindings.samplers[spr.smpUSampler] = state.shadowSampler
  
  sg.applyPipeline(state.pipSprite)
  sg.applyBindings(bindings)
  sg.applyUniforms(spr.ubVsParams, sg.Range(addr: vsParams.addr, size: vsParams.sizeof))
  sg.applyUniforms(spr.ubFsParams, sg.Range(addr: fsParams.addr, size: fsParams.sizeof))
  sg.draw(0, 6, 1)

proc restartLevel(state: var State) =
  state.player.position = vec3(0.0, 12, 25.0)
  state.player.velocity = vec3(0, 0, 0)
  state.player.yaw = 0.0
  state.player.angularVelocity = 0.0
  state.player.rotation = rotate(state.player.yaw, vec3(0, 1, 0))
  state.currentCheckpointIdx = 0
  state.lapCount = 0
  state.lapStartTime = state.time
  state.lastLapTime = 0.0
  state.replayBuffer = @[]
  state.isReplaying = false
  echo "Level Restarted"

proc loadLevel*(state: var State, fs: var RuntimeFS, mapDir: string) =
  # Clear level-specific resources if any
  # (For now, let's just clear the meshes Table and recreate them)
  # Actually, clearResources is currently destructive to everything.
  # We'll just destroy the specific track meshes for now if they exist.
  
  echo &"Loading level: {mapDir}"
  
  let aoParams = AOBakeParams(numRays: 64, maxDistance: 2.0, intensity: 1.0, bias: 0.001)
  let pointSmp = sg.makeSampler(sg.SamplerDesc(minFilter: filterNearest, magFilter: filterNearest))
  
  # Load textures
  let trackTexture1 = loadTexture(state, fs, mapDir/"track_road.qoi")
  let trackTexture2 = loadTexture(state, fs, mapDir/"track_shape.qoi")
  let trackTexture3 = loadTexture(state, fs, mapDir/"track_trees.qoi")
  
  # Load meshes
  state.trackMesh1 = loadAndProcessMesh(state, fs, mapDir/"track_road.ply", aoParams, trackTexture1, pointSmp)
  state.trackMesh2 = loadAndProcessMesh(state, fs, mapDir/"track_shape.ply", aoParams, trackTexture2, pointSmp)
  state.trackMesh3 = loadAndProcessMesh(state, fs, mapDir/"track_trees.ply", aoParams, trackTexture3, pointSmp)
  state.trackMesh4 = loadAndProcessMesh(state, fs, mapDir/"track_barrier.ply", aoParams, trackTexture1, pointSmp)
  
  # Load collision
  (state.roadCollisionVertices, state.roadCollisionIndices) = loadAndProcessMeshCollision(fs, mapDir/"track_road.ply")
  state.roadGrid = initUniformGrid(state.roadCollisionVertices, 64)
  populateGrid(state.roadGrid, state.roadCollisionVertices, state.roadCollisionIndices)
  
  (state.barrierCollisionVertices, state.barrierCollisionIndices) = loadAndProcessMeshCollision(fs, mapDir/"track_barrier.ply")
  state.barrierGrid = initUniformGrid(state.barrierCollisionVertices, 64)
  populateGrid(state.barrierGrid, state.barrierCollisionVertices, state.barrierCollisionIndices)
  
  # Gameplay systems initialization
  state.pathNodes = extractPathFromRoadMesh(state.roadCollisionVertices, state.roadCollisionIndices)
  state.checkpoints = @[]
  for i in countup(0, state.pathNodes.len - 1, 4):
    state.checkpoints.add(Checkpoint(pos: state.pathNodes[i], radius: 18.0))
  
  restartLevel(state)

proc init() {.cdecl.} =
  ASSETS_FS = newRuntimeFS("assets")
  state.gameHasFocus = not defined(emscripten)
  sg.setup(sg.Desc(
    environment: sglue.environment(),
    logger: sg.Logger(fn: slog.fn),
  ))
  audioInit()
  sdtx.setup(sdtx.Desc(
    logger: sdtx.Logger(fn: slog.fn),
    fonts: [ sdtx.fontKc853() ]
  ))
  state.pip = sg.makePipeline(PipelineDesc(
    shader: sg.makeShader(shd.texcubeShaderDesc(sg.queryBackend())),
    layout: VertexLayoutState(
      attrs: [
        VertexAttrState(format: vertexFormatFloat3),
        VertexAttrState(format: vertexFormatFloat3),
        VertexAttrState(format: vertexFormatUbyte4n),
        VertexAttrState(format: vertexFormatFloat2),
        VertexAttrState(format: vertexFormatFloat3),
      ],
    ),
    indexType: indexTypeUint16,
    cullMode: cullModeNone,
    depth: DepthState(
      compare: compareFuncLessEqual,
      writeEnabled: true,
    )
  ))
  initSpritePipeline()
  initShadowTexture()
  initParticleTexture()
  initQuadBuffers()
  initOffscreen()
  initPostfxPipeline()
  initScreenQuad()
  randomize()
  
  state.aoShadowStrength = 1.0
  state.skyLightColor = vec3(0.4, 0.5, 0.8)
  state.skyLightIntensity = 0.0
  state.groundLightColor = vec3(0.6, 0.4, 0.3)
  state.groundLightIntensity = 0.0
  
  loadMusicPlaylist(ASSETS_FS, "music")
  
  let carTexture = loadTexture(state, ASSETS_FS, "car"/"trueno.qoi")
  let aoParams = AOBakeParams(numRays: 64, maxDistance: 2.0, intensity: 1.0, bias: 0.001)
  let pointSmp = sg.makeSampler(sg.SamplerDesc(minFilter: filterNearest, magFilter: filterNearest))
  
  state.carMesh1 = loadAndProcessMesh(state, ASSETS_FS, "car"/"trueno.ply", aoParams, carTexture, pointSmp)
  state.carMesh2 = loadAndProcessMesh(state, ASSETS_FS, "car"/"trueno_back.ply", aoParams, carTexture, pointSmp)
  state.carMesh3 = loadAndProcessMesh(state, ASSETS_FS, "car"/"trueno_front.ply", aoParams, carTexture, pointSmp)
  
  loadLevel(state, ASSETS_FS, "map2")
  
  state.cameraMode = CameraMode.Follow
  state.cameraPos = vec3(0.0, 10.0, 2.0)
  state.cameraOffsetY = 5.0
  state.cameraTarget = state.player.position
  state.lastEmitPos = state.player.position

  state.gameState = GameState.MainMenu
  state.menu.selectedItem = 0
  state.menu.itemCount = 3 # START, CONTROLS, QUIT

proc frame() {.cdecl.} =
  let dt = sapp.frameDuration()
  state.time += dt
  
  if not state.gameHasFocus:
    sg.beginPass(Pass(action: passAction, swapchain: sglue.swapchain()))
    sdtx.canvas(sapp.widthf()*0.5, sapp.heightf()*0.5)
    sdtx.origin(2.0, 2.0)
    sdtx.color3f(1.0, 1.0, 1.0)
    sdtx.puts("CLICK TO PLAY")
    sdtx.draw()
    sg.endPass()
    sg.commit()
    return

  if state.gameState == GameState.Playing:
    # ... physics and logic ...
    block VehiclePhysics:
      const engineForce = 25.0
      const brakeForce = 20.0
      const drag = 0.8
      const angularDrag = 1.1
      const baseGrip = 0.95
      const driftGripMultiplier = 0.3
      const driftTurningMultiplier = 2.2
      const lowSpeedTurnTorque = 100.0
      const highSpeedTurnTorque = 40.0
      const speedForMaxTurnDampening = 40.0
      let prevVelocity = state.player.velocity
      let forwardDir = state.player.rotation * vec3(0, 0, -1)
      if state.input.accelerate:
        state.player.velocity += forwardDir * engineForce * dt
      if state.input.brake:
        if len(state.player.velocity) > 5.0:
          state.player.velocity -= norm(state.player.velocity) * brakeForce * dt
        else:
          state.player.velocity -= forwardDir * engineForce * dt
      let currentSpeed1 = len(state.player.velocity)
      let turnDampeningFactor = clamp(currentSpeed1 / speedForMaxTurnDampening, 0.0, 1.0)
      var effectiveTurningTorque = lerp(lowSpeedTurnTorque, highSpeedTurnTorque, turnDampeningFactor)
      if state.input.drift:
        effectiveTurningTorque *= driftTurningMultiplier
      if state.input.turnLeft:
        state.player.angularVelocity += effectiveTurningTorque * dt
      if state.input.turnRight:
        state.player.angularVelocity -= effectiveTurningTorque * dt
      var currentDrag = drag
      if state.input.drift:
        currentDrag *= 1.5
      state.player.velocity = state.player.velocity * (1.0 - (currentDrag * dt))
      state.player.angularVelocity = state.player.angularVelocity * (1.0 - (angularDrag * dt))
      state.player.yaw += state.player.angularVelocity * dt
      var nextPosition = state.player.position + state.player.velocity * dt
      state.player.rotation = rotate(state.player.yaw, vec3(0, 1, 0))
      let collisionInfo = checkBarrierCollisions(state, nextPosition, state.player.rotation)
      if collisionInfo.collided:
        nextPosition += collisionInfo.pushOut
        # Manual velocity reflection
        let wallNormal = norm(collisionInfo.pushOut)
        let velAlongNormal = dot(state.player.velocity, wallNormal)
        if velAlongNormal < 0:
          state.player.velocity = state.player.velocity * 0.95
          state.player.velocity -= wallNormal * velAlongNormal * 1.05
      state.player.position = nextPosition
      var surfaceUp = vec3.up()
      let surfaceInfoOpt = getSurfaceInfo(state, state.player.position)
      if surfaceInfoOpt.isSome:
        let surfaceInfo = surfaceInfoOpt.get()
        state.player.position.y = surfaceInfo.pos.y + 0.9
        surfaceUp = surfaceInfo.normal
      # Respawn if falling or OOB
      if state.player.position.y < -50.0 or surfaceInfoOpt.isNone:
        # Find last checkpoint passed
        let lastCpIdx = (state.currentCheckpointIdx + state.checkpoints.len - 1) mod state.checkpoints.len
        let respawnPos = state.checkpoints[lastCpIdx].pos
        state.player.position = respawnPos + vec3(0, 2, 0)
        state.player.velocity = vec3(0, 0, 0)
        # Orient toward next checkpoint
        let toNext = norm(state.checkpoints[state.currentCheckpointIdx].pos - respawnPos)
        state.player.yaw = (arctan2(toNext.x, toNext.z) + PI) * (180.0 / PI)
        state.player.rotation = rotate(state.player.yaw, vec3(0, 1, 0))


      let currentSpeed = len(state.player.velocity)
      let carAccel = (currentSpeed - len(prevVelocity)) / dt
      updateEngineSound(currentSpeed, carAccel, state.input.drift, state.debugSpeed, state.debugRpm, state.debugGear)
      var currentGrip = baseGrip
      if state.input.drift:
        currentGrip *= driftGripMultiplier
      let velocityDirection = if currentSpeed > 0.01: norm(state.player.velocity) else: forwardDir
      let newVelocityDir = norm(lerpV(velocityDirection, forwardDir, clamp(currentGrip * dt, 0.0, 1.0)))
      if currentSpeed > 0.01:
        state.player.velocity = newVelocityDir * currentSpeed
      else:
        state.player.velocity = vec3(0,0,0)
      let prevForward = state.player.rotation * vec3(0, 0, -1)
      let yawRot = rotate(state.player.angularVelocity * dt, surfaceUp)
      let newForward = yawRot * prevForward
      let newRight = norm(cross(newForward, surfaceUp))
      let finalForward = norm(cross(surfaceUp, newRight))
      state.player.rotation = fromCols(newRight, surfaceUp, finalForward, vec3(0,0,0))
      
      if state.input.drift:
        let carRot = state.player.rotation
        let distMoved = len(state.player.position - state.lastEmitPos)
        let numSteps = clamp(int(distMoved / 0.05), 1, 15) # Smaller steps for better density
        for s in 0 ..< numSteps:
          let t = s.float / numSteps.float
          let lerpPos = vec3.lerpV(state.lastEmitPos, state.player.position, t)
          let leftEmitPos = lerpPos + (carRot * vec3(-0.8, -0.5, 1.2))
          let rightEmitPos = lerpPos + (carRot * vec3(0.8, -0.5, 1.2))
          let smokeVel = (carRot * vec3(0, 0.5, 1.0)) * 2.0
          let randVel = vec3(rand(-0.5..0.5), rand(0.0..0.5), rand(-0.5..0.5))
          emitParticle(leftEmitPos, smokeVel + randVel, packColor(200, 200, 200, 150), 1.0)
          emitParticle(rightEmitPos, smokeVel + randVel, packColor(200, 200, 200, 150), 1.0)
      
      let exhaustPos = state.player.position + (state.player.rotation * vec3(-0.6, -0.4, 1.5))
      let exhaustVel = (state.player.rotation * vec3(0, 0.2, 0.5)) + vec3(rand(-0.1..0.1), rand(0.1..0.2), rand(-0.1..0.1))
      emitParticle(exhaustPos, exhaustVel, packColor(150, 150, 150, 100), 0.5)
      state.lastEmitPos = state.player.position

    # Phase 3: Gameplay Systems Update
    block GameplayLogic:
      if state.checkpoints.len > 0:
        let nextCp = state.checkpoints[state.currentCheckpointIdx]
        if len(state.player.position - nextCp.pos) < nextCp.radius:
          state.currentCheckpointIdx = (state.currentCheckpointIdx + 1) mod state.checkpoints.len
          if state.currentCheckpointIdx == 0:
            # Completed a lap
            state.lapCount += 1
            state.lastLapTime = state.time - state.lapStartTime
            if state.bestLapTime == 0.0 or state.lastLapTime < state.bestLapTime:
              state.bestLapTime = state.lastLapTime
            state.lapStartTime = state.time

      # Replay Recording / Playback
      if state.isReplaying:
        if state.replayBuffer.len > 0:
          let frame = state.replayBuffer[state.replayIndex]
          state.player.position = frame.pos
          state.player.yaw = frame.yaw
          state.player.rotation = rotate(state.player.yaw, vec3(0, 1, 0))
          state.replayIndex = (state.replayIndex + 1) mod state.replayBuffer.len
      else:
        state.replayBuffer.add(ReplayFrame(pos: state.player.position, yaw: state.player.yaw))
        if state.replayBuffer.len > 10000: # Limit buffer size
          state.replayBuffer.delete(0)

    updateParticles(dt)
    if state.gameState == GameState.Playing:
      updateCamera(state, dt)
    elif state.gameState == GameState.MainMenu:
      # Simple rotating camera for Main Menu
      let radius = 15.0f
      let speed = 0.2f
      state.cameraPos = state.player.position + vec3(cos(state.time * speed) * radius, 5.0, sin(state.time * speed) * radius)
      state.cameraTarget = state.player.position
    
    audioGenerateSamples()

  let fsParams = computeFsParams()
  let proj = persp(60.0f, sapp.widthf() / sapp.heightf(), 0.01f, 150.0f)
  let view = lookat(state.cameraPos, state.cameraTarget, vec3.up())
  
  var offscreenPass = Pass(action: state.offscreenPassAction, attachments: state.offscreenAttachments)
  sg.beginPass(offscreenPass)
  sg.applyPipeline(state.pip)
  sg.applyUniforms(shd.ubFsParams, sg.Range(addr: fsParams.addr, size: fsParams.sizeof))
  let trackModel = identity()
  var trackVsParams = shd.VsParams(u_mvp: proj * view * trackModel, u_model: trackModel, u_camPos: state.cameraPos, u_jitterAmount: 240.0)
  let camForward = norm(state.cameraTarget - state.cameraPos)
  if isMeshVisible(state.trackMesh1, state.cameraPos, camForward):
    sg.applyBindings(state.trackMesh1.bindings)
    sg.applyUniforms(shd.ubVsParams, sg.Range(addr: trackVsParams.addr, size: trackVsParams.sizeof))
    sg.draw(0, state.trackMesh1.indexCount, 1)
  if isMeshVisible(state.trackMesh2, state.cameraPos, camForward):
    sg.applyBindings(state.trackMesh2.bindings)
    sg.applyUniforms(shd.ubVsParams, sg.Range(addr: trackVsParams.addr, size: trackVsParams.sizeof))
    sg.draw(0, state.trackMesh2.indexCount, 1)
  if isMeshVisible(state.trackMesh3, state.cameraPos, camForward):
    sg.applyBindings(state.trackMesh3.bindings)
    sg.applyUniforms(shd.ubVsParams, sg.Range(addr: trackVsParams.addr, size: trackVsParams.sizeof))
    sg.draw(0, state.trackMesh3.indexCount, 1)
 
  drawShadow(proj, view)
  drawCheckpoints(proj, view)
  
  drawParticles(proj, view)
  
  sg.applyPipeline(state.pip)
  # Draw Player
  if state.cameraMode == CameraMode.Follow:
    let carModel = translate(state.player.position) * state.player.rotation
    var carVsParams = shd.VsParams(u_mvp: proj * view * carModel, u_model: carModel, u_camPos: state.cameraPos, u_jitterAmount: sapp.heightf())
    sg.applyBindings(state.carMesh1.bindings)
    sg.applyUniforms(shd.ubVsParams, sg.Range(addr: carVsParams.addr, size: carVsParams.sizeof))
    sg.draw(0, state.carMesh1.indexCount, 1)
    sg.applyBindings(state.carMesh2.bindings)
    sg.applyUniforms(shd.ubVsParams, sg.Range(addr: carVsParams.addr, size: carVsParams.sizeof))
    sg.draw(0, state.carMesh2.indexCount, 1)
    sg.applyBindings(state.carMesh3.bindings)
    sg.applyUniforms(shd.ubVsParams, sg.Range(addr: carVsParams.addr, size: carVsParams.sizeof))
    sg.draw(0, state.carMesh3.indexCount, 1)

  sg.endPass()
  
  sg.beginPass(Pass(action: passAction, swapchain: sglue.swapchain()))
  drawPostfx()
  
  # --- UI ---
  let canvasW = 320.0f
  let canvasH = 240.0f
  sdtx.canvas(canvasW, canvasH)
  let gridW = 40.0f
  let gridH = 30.0f

  if state.gameState == GameState.MainMenu:
    let menuX = 12.0f
    let menuY = 10.0f
    sdtx.pos(menuX, menuY - 4)
    sdtx.color3f(0.2, 0.8, 1.0)
    sdtx.puts("=== PS1 SOKOL RACER ===")
    
    let items = ["START GAME", "CONTROLS", "QUIT"]
    for i, item in items:
      sdtx.pos(menuX, menuY + i.float * 2.0)
      if i == state.menu.selectedItem:
        sdtx.color3f(1.0, 1.0, 1.0)
        sdtx.puts(&"> {item}")
      else:
        sdtx.color3f(0.5, 0.5, 0.5)
        sdtx.puts(&"  {item}")

  elif state.gameState == GameState.Paused:
    let menuX = 14.0f
    let menuY = 10.0f
    sdtx.pos(menuX, menuY - 2)
    sdtx.color3f(1.0, 1.0, 0.0)
    sdtx.puts("=== PAUSED ===")
    
    let items = ["RESUME", "RESTART", "MUSIC VOL", "SFX VOL", "CONTROLS", "QUIT"]
    for i, item in items:
      sdtx.pos(menuX, menuY + i.float * 2.0)
      if i == state.menu.selectedItem:
        sdtx.color3f(1.0, 1.0, 1.0)
        sdtx.puts(&"> {item}")
      else:
        sdtx.color3f(0.5, 0.5, 0.5)
        sdtx.puts(&"  {item}")
      
      if item == "MUSIC VOL":
        sdtx.puts(&" {getMusicVolume()*100:3.0f}%")
      elif item == "SFX VOL":
        sdtx.puts(&" {getSfxVolume()*100:3.0f}%")

  elif state.gameState == GameState.ControlsMenu:
    let menuX = 4.0f
    let menuY = 4.0f
    sdtx.pos(menuX, menuY)
    sdtx.color3f(1.0, 1.0, 0.0)
    sdtx.puts("=== CONTROLS ===")
    
    sdtx.color3f(1.0, 1.0, 1.0)
    let controls = [
      ("W / UP", "ACCELERATE"),
      ("S / DOWN", "BRAKE / REVERSE"),
      ("A / LEFT", "STEER LEFT"),
      ("D / RIGHT", "STEER RIGHT"),
      ("SPACE", "DRIFT"),
      ("R", "RESPAWN AT CHECKPOINT"),
      ("C", "TOGGLE CAMERA"),
      ("ESC / TAB", "PAUSE MENU"),
      ("M", "TOGGLE MUSIC"),
      ("N / B", "NEXT / PREV TRACK"),
      ("P", "TOGGLE REPLAY"),
    ]
    
    for i, ctrl in controls:
      sdtx.pos(menuX, menuY + 2.0 + i.float * 1.5)
      sdtx.color3f(0.2, 0.8, 1.0)
      sdtx.puts(ctrl[0].cstring)
      sdtx.pos(menuX + 12.0, menuY + 2.0 + i.float * 1.5)
      sdtx.color3f(1.0, 1.0, 1.0)
      sdtx.puts(ctrl[1].cstring)
    
    sdtx.pos(menuX, menuY + 20.0)
    sdtx.color3f(0.5, 0.5, 0.5)
    sdtx.puts("PRESS ENTER TO GO BACK")

  if state.gameState != GameState.MainMenu and state.gameState != GameState.ControlsMenu:
    # --- TOP LEFT: MISSION CONTROL ---
    # Checkpoint progress bar
    sdtx.pos(1, 1)
    sdtx.color3f(0.2, 1.0, 0.4)
    let cpTotal = state.checkpoints.len
    if cpTotal > 0:
      let cpCurrent = state.currentCheckpointIdx + 1
      var cpBar = ""
      let barMax = 15
      for i in 1..barMax:
        let progress = i.float / barMax.float
        let cpProgress = cpCurrent.float / cpTotal.float
        if progress < cpProgress: cpBar.add("=")
        elif progress - (1.0/barMax.float) < cpProgress: cpBar.add(">")
        else: cpBar.add("-")
      sdtx.puts(&"POS [{cpBar}] {cpCurrent}/{cpTotal}")

    sdtx.pos(1, 2)
    sdtx.color3f(0.2, 0.8, 1.0)
    sdtx.puts(">> RACE DATA")
    sdtx.color3f(1.0, 1.0, 1.0)
    sdtx.pos(1, 3)
    sdtx.puts(&"LAP  {state.lapCount+1:02}")
    sdtx.pos(1, 4)
    let currentLapTime = state.time - state.lapStartTime
    sdtx.puts(&"TIME {currentLapTime:5.2f}")
    sdtx.pos(1, 5)
    sdtx.puts(&"BEST {state.bestLapTime:5.2f}")

    # --- TOP RIGHT: AUDIO SYSTEM (Anchored to Right) ---
    let audioX = gridW - 16.0
    sdtx.pos(audioX, 2)
    sdtx.color3f(1.0, 0.6, 0.0)
    sdtx.puts(">> AUDIO SYSTEM")
    sdtx.color3f(1.0, 1.0, 1.0)
    sdtx.pos(audioX, 3)
    let trackName = getCurrentTrackFilename()
    let displayName = if trackName.len > 14: trackName[0..11] & ".." else: trackName
    sdtx.puts(displayName.cstring)
    sdtx.pos(audioX, 4)
    sdtx.puts(&"VOL {getMusicVolume()*100:3.0f}%")

    # --- BOTTOM RIGHT: VEHICLE TELEMETRY (Anchored to Bottom-Right) ---
    let dashX = gridW - 16.0
    let dashY = gridH - 5.0
    
    sdtx.pos(dashX, dashY)
    sdtx.color3f(1.0, 0.2, 0.2)
    sdtx.puts(">> TELEMETRY")
    
    # Speed
    sdtx.color3f(1.0, 1.0, 1.0)
    sdtx.pos(dashX, dashY + 1)
    sdtx.puts(&"SPD {state.debugSpeed:5.1f} KM/H")
    
    # RPM
    sdtx.pos(dashX, dashY + 2)
    let rpm = state.debugRpm
    let rpmPct = clamp((rpm - 1000.0) / 5000.0, 0.0, 1.0)
    let rpmBars = int(rpmPct * 10)
    var rpmStr = ""
    for i in 0..<10:
      if i < rpmBars: rpmStr.add("|")
      else: rpmStr.add(".")
    
    if rpmPct > 0.8: sdtx.color3f(1.0, 0.0, 0.0) # Redline
    elif rpmPct > 0.5: sdtx.color3f(1.0, 1.0, 0.0) # Mid
    else: sdtx.color3f(0.0, 1.0, 0.0) # Low
    sdtx.puts(&"RPM [{rpmStr}]")
    
    # Gear
    sdtx.color3f(1.0, 1.0, 1.0)
    sdtx.pos(dashX, dashY + 3)
    sdtx.puts(&"GEAR {state.debugGear}")

  sdtx.draw()
  sg.endPass()
  sg.commit()

proc cleanup() {.cdecl.} =
  clearResources(state)
  sdtx.shutdown()
  audioShutdown()
  sg.shutdown()

proc event(e: ptr sapp.Event) {.cdecl.} =
  if e.`type` == EventType.eventTypeFocused: state.gameHasFocus = true
  elif e.`type` == EventType.eventTypeUnfocused: state.gameHasFocus = false
  if e.`type` == EventType.eventTypeMouseDown:
    if not state.gameHasFocus:
      state.gameHasFocus = true
      when defined(emscripten): emscripten_run_script("document.getElementById('canvas').focus();")
  
  if e.`type` == EventType.eventTypeMouseScroll:
    state.cameraOffsetY += e.scrollY * 0.5
    state.cameraOffsetY = max(state.cameraOffsetY, 0.0)

  if e.`type` == EventType.eventTypeKeyDown:
    case e.keyCode
    of keyCodeEscape, keyCodeTab:
      if state.gameState == GameState.Playing:
        state.gameState = GameState.Paused
        state.menu.selectedItem = 0
        state.menu.itemCount = 6 # RESUME, RESTART, MUSIC, SFX, CONTROLS, QUIT
      elif state.gameState == GameState.Paused or state.gameState == GameState.ControlsMenu:
        state.gameState = GameState.Playing
    of keyCodeW, keyCodeUp:
      if state.gameState == GameState.Paused or state.gameState == GameState.MainMenu:
        state.menu.selectedItem = (state.menu.selectedItem + state.menu.itemCount - 1) mod state.menu.itemCount
    of keyCodeS, keyCodeDown:
      if state.gameState == GameState.Paused or state.gameState == GameState.MainMenu:
        state.menu.selectedItem = (state.menu.selectedItem + 1) mod state.menu.itemCount
    of keyCodeA, keyCodeLeft:
      if state.gameState == GameState.Paused:
        if state.menu.selectedItem == 2: # MUSIC VOLUME
          setMusicVolume(getMusicVolume() - 0.1)
        elif state.menu.selectedItem == 3: # SFX VOLUME
          setSfxVolume(getSfxVolume() - 0.1)
    of keyCodeD, keyCodeRight:
      if state.gameState == GameState.Paused:
        if state.menu.selectedItem == 2: # MUSIC VOLUME
          setMusicVolume(getMusicVolume() + 0.1)
        elif state.menu.selectedItem == 3: # SFX VOLUME
          setSfxVolume(getSfxVolume() + 0.1)
    of keyCodeEnter, keyCodeSpace:
      if state.gameState == GameState.MainMenu:
        case state.menu.selectedItem
        of 0: state.gameState = GameState.Playing # START
        of 1:
          state.previousGameState = state.gameState
          state.gameState = GameState.ControlsMenu # CONTROLS
        of 2: sapp.requestQuit() # QUIT
        else: discard
      elif state.gameState == GameState.Paused:
        case state.menu.selectedItem
        of 0: state.gameState = GameState.Playing # RESUME
        of 1: # RESTART
          restartLevel(state)
          state.gameState = GameState.Playing
        of 2, 3: discard # VOLUME (handled by left/right)
        of 4:
          state.previousGameState = state.gameState
          state.gameState = GameState.ControlsMenu # CONTROLS
        of 5: sapp.requestQuit() # QUIT
        else: discard
      elif state.gameState == GameState.ControlsMenu:
        state.gameState = state.previousGameState # Go back to menu
    else: discard

  if e.`type` == EventType.eventTypeKeyDown or e.`type` == EventType.eventTypeKeyUp:
    let step: float32 = 0.05
    let isDown = e.`type` == EventType.eventTypeKeyDown
    
    # Game Controls (only if playing)
    if state.gameState == GameState.Playing:
      case e.keyCode
      of keyCodeW: state.input.accelerate = isDown
      of keyCodeS: state.input.brake = isDown
      of keyCodeA: state.input.turnLeft = isDown
      of keyCodeD: state.input.turnRight = isDown
      of keyCodeSpace: state.input.drift = isDown
      of keyCodeR:
        if isDown:
          let lastCpIdx = (state.currentCheckpointIdx + state.checkpoints.len - 1) mod state.checkpoints.len
          let respawnPos = state.checkpoints[lastCpIdx].pos
          state.player.position = respawnPos + vec3(0, 2, 0)
          state.player.velocity = vec3(0, 0, 0)
          let toNext = state.checkpoints[state.currentCheckpointIdx].pos - respawnPos
          state.player.yaw = (arctan2(toNext.x, toNext.z) + PI) * (180.0 / PI)
          state.player.rotation = rotate(state.player.yaw, vec3(0, 1, 0))
      of keyCodeC:
        if isDown:
          state.cameraMode = if state.cameraMode == CameraMode.Follow: CameraMode.Front else: CameraMode.Follow
      of keyCode1: state.aoShadowStrength = max(0.0, state.aoShadowStrength - step)
      of keyCode2: state.aoShadowStrength += step
      of keyCode3: state.skyLightIntensity = max(0.0, state.skyLightIntensity - step)
      of keyCode4: state.skyLightIntensity += step
      of keyCode5: state.groundLightIntensity = max(0.0, state.groundLightIntensity - step)
      of keyCode6: state.groundLightIntensity += step
      of keyCodeP:
        if isDown:
          state.isReplaying = not state.isReplaying
          state.replayIndex = 0
          if state.isReplaying:
            state.input = InputState() # Clear input
      of keyCodeN:
        if isDown: nextTrack()
      of keyCodeB:
        if isDown: prevTrack()
      of keyCodeM:
        if isDown: toggleMusic()
      of keyCode9:
        if isDown: setMusicVolume(getMusicVolume() - 0.1)
      of keyCode0:
        if isDown: setMusicVolume(getMusicVolume() + 0.1)
      else: discard

sapp.run(sapp.Desc(
  initCb: init,
  frameCb: frame,
  eventCb: event,
  cleanupCb: cleanup,
  windowTitle: "Game",
  width: 640,
  height: 480,
  sampleCount: 0,
  icon: IconDesc(sokol_default: true),
  logger: sapp.Logger(fn: slog.fn)
))
