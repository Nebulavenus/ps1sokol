import sokol/log as slog
import sokol/app as sapp
import sokol/gfx as sg
import sokol/glue as sglue
import shaders/default as shd
import math/vec2
import math/vec3
import math/mat4
import math
import strutils
import tables
import os

type Vertex = object
  x, y, z: float32
  xN, yN, zN: float32
  color: uint32
  u, v: uint16

type Mesh = object
  bindings: Bindings
  indexCount: int32

proc vec2ToUV(v: Vec2): (uint16, uint16) =
  # Define the min and max values for normalization
  const
    minX = 0.0
    maxX = 1.0
    minY = 0.0
    maxY = 1.0

  # Normalize the Vec2 values to the range [0, 1]
  let normalizedX = (v.x - minX) / (maxX - minX)
  let normalizedY = (v.y - minY) / (maxY - minY)

  # Clamp the values to ensure they are within [0, 1]
  let clampedX = clamp(normalizedX, 0.0, 1.0)
  let clampedY = clamp(normalizedY, 0.0, 1.0)

  # Scale to uint16 range
  let uvX = cast[uint16](clampedX * 65535.0)
  let uvY = cast[uint16](clampedY * 65535.0)

  return (uvX, uvY)

proc loadObj(path: string): Mesh =
  var
    temp_positions: seq[Vec3]
    temp_normals: seq[Vec3]
    temp_uvs: seq[Vec2]
    out_vertices: seq[Vertex]
    out_indices: seq[uint16]
    vertex_cache: Table[string, uint16]

  if not os.fileExists(path):
    echo("Cannot load OBJ file, path not found: " & path)
    return

  echo "Starting OBJ Loading"
  for line in lines(path):
    if line.startsWith("v "):
      let parts = line.split()
      temp_positions.add(vec3(
        parts[1].parseFloat,
        parts[2].parseFloat,
        parts[3].parseFloat,
      ))
    elif line.startsWith("vn "):
      let parts = line.split()
      temp_normals.add(vec3(
        parts[1].parseFloat,
        parts[2].parseFloat,
        parts[3].parseFloat,
      ))
    elif line.startsWith("vt "):
      let parts = line.split()
      temp_uvs.add(vec2(
        parts[1].parseFloat,
        parts[2].parseFloat,
      ))
    elif line.startsWith("f "):
      let face_parts = line.split()
      for i in 1..3: # For each vertex in the triangle face
        let key = face_parts[i]

        if not vertex_cache.haskey(key):
          var
            pos_idx = -1
            uv_idx = -1
            nrm_idx = -1

          let v_parts = key.split('/')

          # Parse based on the number of components found
          case v_parts.len
          of 1: # Format "v"
            pos_idx = v_parts[0].parseInt - 1
          of 2: # Format "v/vt"
            pos_idx = v_parts[0].parseInt - 1
            uv_idx = v_parts[1].parseInt - 1
          of 3: # Format "v/vt/vn" or "v//vn"
            pos_idx = v_parts[0].parseInt - 1
            if v_parts[1].len > 0: # Check if vt is present
              uv_idx = v_parts[1].parseInt - 1
            nrm_idx = v_parts[2].parseInt - 1
          else:
            echo("Unsupported face format component: " & key)
            continue

          # Create the vertex, providing defaults for missing data
          let pos = if pos_idx != -1: temp_positions[pos_idx] else: vec3(0,0,0)
          let nrm = if nrm_idx != -1: temp_normals[nrm_idx] else: vec3(0,1,0) # Default normal points up
          let uv  = if uv_idx != -1: temp_uvs[uv_idx] else: vec2(0,0) # Default UV is 0,0
          let uvS = vec2ToUV(uv)
          # Obj doesn't store vertex colors... by default white
          let new_vert = Vertex(
            x: pos.x, y: pos.y, z: pos.z,
            xN: nrm.x, yN: nrm.y, zN: nrm.z,
            color: 0xFFFFFFFF'u32,
            u: uvS[0], v: uvS[1]
          )
          out_vertices.add(new_vert)
          let new_idx = (out_vertices.len - 1).uint16
          vertex_cache[key] = new_idx
          out_indices.add(new_idx)
        else:
          # Vertex already exists, just add its index
          out_indices.add(vertex_cache[key])

  echo "Loaded OBJ $1: $2 vertices, $3 indices" % [$path, $out_vertices.len, $out_indices.len]
  let vbuf = sg.makeBuffer(BufferDesc(
    usage: BufferUsage(vertexBuffer: true),
    data: sg.Range(addr: out_vertices[0].addr, size: out_vertices.len * sizeof(Vertex))
  ))
  let ibuf = sg.makeBuffer(BufferDesc(
    usage: BufferUsage(indexBuffer: true),
    data: sg.Range(addr: out_indices[0].addr, size: out_indices.len * sizeof(uint16))
  ))
  result.indexCount = out_indices.len.int32
  result.bindings = Bindings(vertexBuffers: [vbuf], indexBuffer: ibuf)

type State = object
  pip: Pipeline
  passAction: sg.PassAction
  mesh: Mesh
  camTime: float32
  camPos: Vec3
  camYaw: float32
  camPitch: float32
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

  sapp.lockMouse(true)
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
  #mesh.bindings = Bindings(vertexBuffers: [vbuf], indexbuffer: ibuf)
  #mesh.bindings.images[shd.imgUTexture] = texcubeImg
  #mesh.bindings.samplers[shd.smpUSampler] = texcubeSmp
  #mesh.indexCount = indices.sizeof
  #state.mesh = mesh

  let assetDir = getAppDir() & DirSep
  #let modelPath = assetDir & "bs_rest.obj"
  let modelPath = assetDir & "teapot.obj"

  mesh = loadObj(modelPath)
  mesh.bindings.images[shd.imgUTexture] = texcubeImg
  mesh.bindings.samplers[shd.smpUSampler] = texcubeSmp
  state.mesh = mesh

proc computeVsParams(): shd.VsParams =
  let camStart = state.camPos + vec3(0.0, 2.5, 4.0)
  let camEnd = state.camPos + vec3(0.0, 0.5, 12.0)
  let dt = sapp.frameDuration()
  let speed = 0.3 # cycles per second
  state.camTime += dt
  #let t = (1.0 + math.sin(2.0 * 3.14159 * speed * state.camTime)) / 2.0 # move sin [-1,1] to [0, 1]
  #let camPos = camStart + (camEnd - camStart) * t
  let camPos = state.camPos

  let proj = persp(60.0f, sapp.widthf() / sapp.heightf(), 0.01f, 20.0f)

  # Calculate the camera's forward direction vector using spherical coordinates
  let forwardVec = norm(vec3(
    cos(state.camPitch) * sin(state.camYaw),
    sin(state.camPitch),
    cos(state.camPitch) * -cos(state.camYaw)
  ))
  let lookAtPoint = camPos + forwardVec
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
  #state.rx += 1f * dt
  #state.ry += 2f * dt

  let vsParams = computeVsParams()
  let fsParams = computeFsParams()

  sg.beginPass(Pass(action: passAction, swapchain: sglue.swapchain()))
  sg.applyPipeline(state.pip)
  sg.applyBindings(state.mesh.bindings)
  sg.applyUniforms(shd.ubVsParams, sg.Range(addr: vsParams.addr, size: vsParams.sizeof))
  sg.applyUniforms(shd.ubFsParams, sg.Range(addr: fsParams.addr, size: fsParams.sizeof))
  sg.draw(0, state.mesh.indexCount, 1)
  sg.endPass()
  sg.commit()

proc cleanup() {.cdecl.} =
  sg.shutdown()

proc event(e: ptr sapp.Event) {.cdecl.} =
  # Mouse
  if e.`type` == EventType.eventTypeMouseMove:
    let mouseSensitivity = 0.005 # Adjust this value to your liking
    state.camYaw   += e.mouseDx * mouseSensitivity
    state.camPitch -= e.mouseDy * mouseSensitivity # Subtract because positive dy is mouse down

    # Clamp pitch to prevent the camera from flipping over
    const pitchLimit = 1.55 # ~89 degrees in radians
    if state.camPitch > pitchLimit: state.camPitch = pitchLimit
    if state.camPitch < -pitchLimit: state.camPitch = -pitchLimit

  # Keyboard
  if e.`type` == EventType.eventTypeKeyDown:
    let moveSpeed = 0.5
    let rotSpeed = 0.3
    let forwardVec = vec3(sin(state.camYaw), 0.0, -cos(state.camYaw))
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
    of keyCodeQ:
      state.camPos.y += moveSpeed
    of keyCodeE:
      state.camPos.y -= moveSpeed
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
