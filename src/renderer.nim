import math
import sokol/gfx as sg
import sokol/app as sapp
import math/vec2
import math/vec3
import math/mat4
import types
import shaders/default as shd
import shaders/sprite as spr
import shaders/postfx as pfx
import options
import culling
import physics
import particles
import colors

proc computeFsParams*(state: var State): shd.FsParams =
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

proc initShadowTexture*(state: var State) =
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

proc initQuadBuffers*(state: var State) =
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

proc initSpritePipeline*(state: var State) =
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
      writeEnabled: false,
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

proc initOffscreen*(state: var State) =
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

proc initScreenQuad*(state: var State) =
  let vertices = [
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

proc initPostfxPipeline*(state: var State) =
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

proc drawPostfx*(state: var State) =
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

proc drawCheckpoints*(state: var State, proj, view: Mat4) =
  if state.checkpoints.len == 0: return
  let cp = state.checkpoints[state.currentCheckpointIdx]
  let camForward = norm(state.cameraTarget - state.cameraPos)
  let camRight = norm(cross(camForward, vec3.up()))
  let cpModel = translate(cp.pos + vec3(0, 5, 0)) * fromCols(camRight * cp.radius, vec3.up() * 10.0, camForward * cp.radius, vec3(0,0,0))
  
  var vsParams = spr.VsParams(u_mvp: proj * view * cpModel, u_camPos: state.cameraPos, u_jitterAmount: 240.0)
  var fsParams = spr.FsParams(u_fogColor: vec3(0.25f, 0.5f, 0.75f), u_fogNear: 50.0f, u_fogFar: 150.0f, u_alphaThreshold: 0.01f)
  
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

proc drawShadow*(state: var State, proj, view: Mat4) =
  let surfaceHitOpt = getSurfaceInfo(state, state.player.position)
  if surfaceHitOpt.isNone: return
  let hit = surfaceHitOpt.get()
  let shadowPos = hit.pos + hit.normal * 0.05
  let carRotationOnlyYaw = rotate(state.player.yaw, vec3(0, 1, 0))
  let shadowModel = translate(shadowPos) * carRotationOnlyYaw * scale(vec3(1.5, 1.0, 2.5))
  
  var vsParams = spr.VsParams(u_mvp: proj * view * shadowModel, u_camPos: state.cameraPos, u_jitterAmount: 240.0)
  var fsParams = spr.FsParams(u_fogColor: vec3(0.25f, 0.5f, 0.75f), u_fogNear: 1000.0f, u_fogFar: 1500.0f, u_alphaThreshold: 0.01f)
  
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

proc drawVehicle*(state: var State, proj, view, model: Mat4, camPos: Vec3, carIdx: int) =
  if carIdx < 0 or carIdx >= state.carMeshes.len: return
  var vsParams = shd.VsParams(u_mvp: proj * view * model, u_model: model, u_camPos: camPos, u_jitterAmount: sapp.heightf())
  for mesh in state.carMeshes[carIdx]:
    sg.applyBindings(mesh.bindings)
    sg.applyUniforms(shd.ubVsParams, sg.Range(addr: vsParams.addr, size: vsParams.sizeof))
    sg.draw(0, mesh.indexCount, 1)
