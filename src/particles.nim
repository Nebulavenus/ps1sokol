import sokol/gfx as sg
import math
import math/vec3
import math/mat4
import std/algorithm
import types
import colors
import shaders/sprite as spr

proc makeCircularTexture*(size: int, alphaMultiplier: float32 = 255.0): sg.Image =
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

proc initParticleTexture*(state: var State) =
  state.particleTexture = makeCircularTexture(32, 255.0)
  state.checkpointTexture = makeCircularTexture(64, 100.0)
  state.nitroTexture = makeCircularTexture(48, 255.0) # Nitro pickup texture

proc emitParticle*(state: var State, pos, vel: Vec3, color: uint32, life: float32) =
  let idx = state.particles.nextIndex
  state.particles.pool[idx] = Particle(
    pos: pos,
    vel: vel,
    color: color,
    life: life,
    maxLife: life
  )
  state.particles.nextIndex = (idx + 1) mod state.particles.pool.len

proc updateParticles*(state: var State, dt: float32) =
  for i in 0 ..< state.particles.pool.len:
    var p = addr state.particles.pool[i]
    if p.life > 0:
      p.life -= dt
      p.pos = p.pos + (p.vel * dt)
      p.vel = p.vel * 0.95 # Air resistance

proc drawParticles*(state: var State, proj, view: Mat4) =
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
