import sokol/log as slog
import sokol/app as sapp
import sokol/gfx as sg
import sokol/glue as sglue
import shaders/default as shd
import math/vec2
import math/vec3
import math/mat4
import math

type Mesh = object
  bindings: Bindings
  indexCount: int

type State = object
  pip: Pipeline
  passAction: sg.PassAction
  mesh: Mesh
  camTime: float32
  camPos: Vec3
  camAngle: float32
  vsParams: VSParams
  fsParams: FSParams
  rx, ry: float32

var state: State

const
  passAction = PassAction(
    colors: [
      ColorAttachmentAction(
        loadAction: loadActionClear,
        clearValue: (0.25, 0.5, 0.75, 1) # same for fog color
      )
    ]
  )

type Vertex = object
  x, y, z: float32
  xN, yN, zN: float32
  color: uint32
  u, v: uint16

proc init() {.cdecl.} =
  sg.setup(sg.Desc(
    environment: sglue.environment(),
    logger: sg.Logger(fn: slog.fn),
  ))
  case sg.queryBackend():
    of backendGlcore: echo "using GLCORE33 backend"
    of backendD3d11: echo "using D3D11 backend"
    of backendMetalMacos: echo "using Metal backend"
    else: echo "using untested backend"

  #[
    Cube vertex buffer with packed vertex formats for color and texture coords.
    Note that a vertex format which must be portable across all
    backends must only use the normalized integer formats
    (BYTE4N, UBYTE4N, SHORT2N, SHORT4N), which can be converted
    to floating point formats in the vertex shader inputs.
  ]#
  const vertices = [
    Vertex( x: -1.0, y: -1.0, z: -1.0,  xN: 0.0, yN: 0.0, zN: -1.0,  color: 0xFF0000FF'u32, u:     0, v:     0 ),
    Vertex( x:  1.0, y: -1.0, z: -1.0,  xN: 0.0, yN: 0.0, zN: -1.0,  color: 0xFF0000FF'u32, u: 32767, v:     0 ),
    Vertex( x:  1.0, y:  1.0, z: -1.0,  xN: 0.0, yN: 0.0, zN: -1.0,  color: 0xFF0000FF'u32, u: 32767, v: 32767 ),
    Vertex( x: -1.0, y:  1.0, z: -1.0,  xN: 0.0, yN: 0.0, zN: -1.0,  color: 0xFF0000FF'u32, u:     0, v: 32767 ),
    Vertex( x: -1.0, y: -1.0, z:  1.0,  xN: 0.0, yN: 0.0, zN: 1.0,  color: 0xFF00FF00'u32, u:     0, v:     0 ),
    Vertex( x:  1.0, y: -1.0, z:  1.0,  xN: 0.0, yN: 0.0, zN: 1.0,  color: 0xFF00FF00'u32, u: 32767, v:     0 ),
    Vertex( x:  1.0, y:  1.0, z:  1.0,  xN: 0.0, yN: 0.0, zN: 1.0,  color: 0xFF00FF00'u32, u: 32767, v: 32767 ),
    Vertex( x: -1.0, y:  1.0, z:  1.0,  xN: 0.0, yN: 0.0, zN: 1.0,  color: 0xFF00FF00'u32, u:     0, v: 32767 ),
    Vertex( x: -1.0, y: -1.0, z: -1.0,  xN: -1.0, yN: 0.0, zN: 0.0,  color: 0xFFFF0000'u32, u:     0, v:     0 ),
    Vertex( x: -1.0, y:  1.0, z: -1.0,  xN: -1.0, yN: 0.0, zN: 0.0,  color: 0xFFFF0000'u32, u: 32767, v:     0 ),
    Vertex( x: -1.0, y:  1.0, z:  1.0,  xN: -1.0, yN: 0.0, zN: 0.0,  color: 0xFFFF0000'u32, u: 32767, v: 32767 ),
    Vertex( x: -1.0, y: -1.0, z:  1.0,  xN: -1.0, yN: 0.0, zN: 0.0,  color: 0xFFFF0000'u32, u:     0, v: 32767 ),
    Vertex( x:  1.0, y: -1.0, z: -1.0,  xN: 1.0, yN: 0.0, zN: 0.0,  color: 0xFFFF007F'u32, u:     0, v:     0 ),
    Vertex( x:  1.0, y:  1.0, z: -1.0,  xN: 1.0, yN: 0.0, zN: 0.0,  color: 0xFFFF007F'u32, u: 32767, v:     0 ),
    Vertex( x:  1.0, y:  1.0, z:  1.0,  xN: 1.0, yN: 0.0, zN: 0.0,  color: 0xFFFF007F'u32, u: 32767, v: 32767 ),
    Vertex( x:  1.0, y: -1.0, z:  1.0,  xN: 1.0, yN: 0.0, zN: 0.0,  color: 0xFFFF007F'u32, u:     0, v: 32767 ),
    Vertex( x: -1.0, y: -1.0, z: -1.0,  xN: 0.0, yN: -1.0, zN: 0.0,  color: 0xFFFF7F00'u32, u:     0, v:     0 ),
    Vertex( x: -1.0, y: -1.0, z:  1.0,  xN: 0.0, yN: -1.0, zN: 0.0,  color: 0xFFFF7F00'u32, u: 32767, v:     0 ),
    Vertex( x:  1.0, y: -1.0, z:  1.0,  xN: 0.0, yN: -1.0, zN: 0.0,  color: 0xFFFF7F00'u32, u: 32767, v: 32767 ),
    Vertex( x:  1.0, y: -1.0, z: -1.0,  xN: 0.0, yN: -1.0, zN: 0.0,  color: 0xFFFF7F00'u32, u:     0, v: 32767 ),
    Vertex( x: -1.0, y:  1.0, z: -1.0,  xN: 0.0, yN: 1.0, zN: 0.0,  color: 0xFF007FFF'u32, u:     0, v:     0 ),
    Vertex( x: -1.0, y:  1.0, z:  1.0,  xN: 0.0, yN: 1.0, zN: 0.0,  color: 0xFF007FFF'u32, u: 32767, v:     0 ),
    Vertex( x:  1.0, y:  1.0, z:  1.0,  xN: 0.0, yN: 1.0, zN: 0.0,  color: 0xFF007FFF'u32, u: 32767, v: 32767 ),
    Vertex( x:  1.0, y:  1.0, z: -1.0,  xN: 0.0, yN: 1.0, zN: 0.0,  color: 0xFF007FFF'u32, u:     0, v: 32767 ),
  ]
  let vbuf = sg.makeBuffer(BufferDesc(
    usage: BufferUsage(vertexBuffer: true),
    data: sg.Range(addr: vertices.addr, size: vertices.sizeof)
  ))

  # an index buffer
  const indices = [
    0'u16, 1, 2,  0, 2, 3,
    6, 5, 4,      7, 6, 4,
    8, 9, 10,     8, 10, 11,
    14, 13, 12,   15, 14, 12,
    16, 17, 18,   16, 18, 19,
    22, 21, 20,   23, 22, 20,
  ]
  let ibuf = sg.makeBuffer(BufferDesc(
    usage: BufferUsage(indexBuffer: true),
    data: sg.Range(addr: indices.addr, size: indices.sizeof)
  ))

  # create a checker board texture
  let pixels = [
    0xFFFFFFFF'u32, 0xFF000000'u32, 0xFFFFFFFF'u32, 0xFF000000'u32,
    0xFF000000'u32, 0xFFFFFFFF'u32, 0xFF000000'u32, 0xFFFFFFFF'u32,
    0xFFFFFFFF'u32, 0xFF000000'u32, 0xFFFFFFFF'u32, 0xFF000000'u32,
    0xFF000000'u32, 0xFFFFFFFF'u32, 0xFF000000'u32, 0xFFFFFFFF'u32,
  ]
  #bindings.images[shd.imgTex] = sg.makeImage(sg.ImageDesc(
  let texcubeImg = sg.makeImage(sg.ImageDesc(
    width: 4,
    height: 4,
    data: ImageData(
      subimage: [ [ sg.Range(addr: pixels.addr, size: pixels.sizeof) ] ]
    )
  ))

  # create a matching sampler
  #bindings.samplers[shd.smpSmp] = sg.makeSampler(sg.SamplerDesc(
  let texcubeSmp = sg.makeSampler(sg.SamplerDesc(
    minFilter: filterNearest,
    magFilter: filterNearest,
  ));

  # create shader and pipeline object
  state.pip = sg.makePipeline(PipelineDesc(
    shader: sg.makeShader(shd.texcubeShaderDesc(sg.queryBackend())),
    layout: VertexLayoutState(
      attrs: [
        VertexAttrState(format: vertexFormatFloat3),  # position
        VertexAttrState(format: vertexFormatFloat3),  # normal
        VertexAttrState(format: vertexFormatUbyte4n), # color0
        VertexAttrState(format: vertexFormatShort2n), # texcoord0
      ],
    ),
    indexType: indexTypeUint16,
    cullMode: cullModeBack,
    depth: DepthState(
      compare: compareFuncLessEqual,
      writeEnabled: true,
    )
  ))
  # save everything in bindings
  var mesh: Mesh
  mesh.bindings = Bindings(vertexBuffers: [vbuf], indexbuffer: ibuf)
  mesh.bindings.images[shd.imgUTexture] = texcubeImg
  mesh.bindings.samplers[shd.smpUSampler] = texcubeSmp
  mesh.indexCount = indices.sizeof
  state.mesh = mesh

proc computeVsParams(): shd.VsParams =
  let camStart = state.camPos + vec3(0.0, 2.5, 4.0)
  let camEnd = state.camPos + vec3(0.0, 0.5, 12.0)
  let dt = sapp.frameDuration()
  let speed = 0.3 # cycles per second
  state.camTime += dt
  let t = (1.0 + math.sin(2.0 * 3.14159 * speed * state.camTime)) / 2.0 # move sin [-1,1] to [0, 1]
  let camPos = camStart + (camEnd - camStart) * t

  let proj = persp(60.0f, sapp.widthf() / sapp.heightf(), 0.01f, 20.0f)

  let lookAtPoint = camPos + vec3(sin(state.camAngle), 0.0, -cos(state.camAngle))
  let view = lookat(camPos, lookAtPoint, vec3.up())

  let rxm = rotate(state.rx, vec3(1f, 0f, 0f))
  let rym = rotate(state.ry, vec3(0f, 1f, 0f))
  let model = rxm * rym
  result = shd.VsParams(
    u_mvp: proj * view * model,
    u_model: model,
    u_camPos: camPos,
    u_jitterAmount: 240.0, # Simulate a 240p vertical resolution
  )

proc computeFsParams(): shd.FsParams =
  result = shd.FsParams(
    u_fogColor: vec3(0.25f, 0.5f, 0.75f),
    u_fogNear: 4.0f,
    u_fogFar: 12.0f,
    #u_ditherSize: vec2(800.0, 600.0)
    u_ditherSize: vec2(sapp.widthf(), sapp.heightf()),
  )

proc frame() {.cdecl.} =
  let dt = sapp.frameDuration() * 60f
  state.rx += 1f * dt
  state.ry += 2f * dt

  let vsParams = computeVsParams()
  let fsParams = computeFsParams()

  sg.beginPass(Pass(action: passAction, swapchain: sglue.swapchain()))
  sg.applyPipeline(state.pip)
  sg.applyBindings(state.mesh.bindings)
  sg.applyUniforms(shd.ubVsParams, sg.Range(addr: vsParams.addr, size: vsParams.sizeof))
  sg.applyUniforms(shd.ubFsParams, sg.Range(addr: fsParams.addr, size: fsParams.sizeof))
  sg.draw(0, 36, 1)
  sg.endPass()
  sg.commit()

proc cleanup() {.cdecl.} =
  sg.shutdown()

proc event(e: ptr sapp.Event) {.cdecl.} =
  if e.`type` == EventType.eventTypeKeyDown:
    let moveSpeed = 0.5
    let rotSpeed = 0.3
    let forwardVec = vec3(sin(state.camAngle), 0.0, -cos(state.camAngle))
    let rightVec = vec3(forwardVec.z, 0.0, -forwardVec.x)

    case e.keyCode
    of keyCodeEscape:
      sapp.requestQuit()
    of keyCodeW:
      state.camPos += (forwardVec * moveSpeed)
    of keyCodeS:
      state.camPos -= forwardVec * moveSpeed
    of keyCodeA:
      state.camPos -= rightVec * moveSpeed
    of keyCodeD:
      state.camPos += rightVec * moveSpeed
    of keyCodeQ, keyCodeLeft:
      state.camAngle -= rotSpeed
    of keyCodeE, keyCodeRight:
      state.camAngle += rotSpeed
    else: discard

# main

sapp.run(sapp.Desc(
  initCb: init,
  frameCb: frame,
  eventCb: event,
  cleanupCb: cleanup,
  windowTitle: "Game",
  width: 640,
  height: 480,
  sampleCount: 4,
  icon: IconDesc(sokol_default: true),
  logger: sapp.Logger(fn: slog.fn)
))
