import types
import sokol/gfx as sg
import math/vec2
import math/vec3
import rtfs
import strutils
import tables
import streams
import options
import os
import qoi
import colors
import aobaker
import shaders/default as shd

proc loadObj*(fileLines: seq[string]): (seq[Vertex], seq[uint16]) =
  var
    temp_positions: seq[Vec3]
    temp_normals: seq[Vec3]
    temp_uvs: seq[Vec2]
    out_vertices: seq[Vertex]
    out_indices: seq[uint16]
    vertex_cache: Table[string, uint16]

  for line in fileLines:
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

          let pos = if pos_idx != -1: temp_positions[pos_idx] else: vec3(0,0,0)
          let nrm = if nrm_idx != -1: temp_normals[nrm_idx] else: vec3(0,1,0)
          var uv  = if uv_idx != -1: temp_uvs[uv_idx] else: vec2(0,0)
          uv.y = 1.0 - uv.y
          let new_vert = Vertex(
            x: pos.x, y: pos.y, z: pos.z,
            xN: nrm.x, yN: nrm.y, zN: nrm.z,
            color: 0xFFFFFFFF'u32,
            u: uv.x, v: uv.y
          )
          out_vertices.add(new_vert)
          let new_idx = (out_vertices.len - 1).uint16
          vertex_cache[key] = new_idx
          out_indices.add(new_idx)
        else:
          out_indices.add(vertex_cache[key])

  echo "Loaded OBJ: $1 vertices, $2 indices" % [$out_vertices.len, $out_indices.len]
  return (out_vertices, out_indices)

proc loadPly*(fileLines: seq[string]): (seq[Vertex], seq[uint16]) =
  var
    out_vertices: seq[Vertex]
    out_indices: seq[uint16]

  var
    vertexCount = 0
    faceCount = 0
    inHeader = true
    vertexPropertyMap: Table[string, int]
    vertexPropertyCount = 0
    parsingVertex = false

  var bodyStartIndex = -1

  for i, line in fileLines:
    if not inHeader: continue
    let parts = line.split()
    if parts.len == 0: continue
    case parts[0]
    of "ply": discard
    of "format":
      if parts.len > 1 and parts[1] == "ascii":
        echo "Ply is in ASCII format"
      else:
        echo "Unsupported or invalid PLY format"
        return
    of "comment": discard
    of "element":
      parsingVertex = false
      if parts.len == 3 and parts[1] == "vertex":
        vertexCount = parts[2].parseInt
        parsingVertex = true
        vertexPropertyCount = 0
      elif parts.len == 3 and parts[1] == "face":
        faceCount = parts[2].parseInt
    of "property":
      if parsingVertex and parts.len == 3:
        var propName = parts[^1]
        if propName == "u": propName = "s"
        if propName == "v": propName = "t"
        vertexPropertyMap[propName] = vertexPropertyCount
        vertexPropertyCount += 1
    of "end_header":
      inHeader = false
      bodyStartIndex = i + 1
      break
    else: discard

  if bodyStartIndex == -1:
    echo "loadPly, Failed to parse PLY header"
    return

  echo "loadPly, Header parsed. Vertices: $1, Faces: $2" % [$vertexCount, $faceCount]

  let vertexLinesEnd = bodyStartIndex + vertexCount
  out_vertices.setLen(vertexCount)
  for i in bodyStartIndex ..< vertexLinesEnd:
    let parts = fileLines[i].split()
    if parts.len != vertexPropertyCount:
      continue

    proc getProp(name: string, default: float): float =
      if vertexPropertyMap.haskey(name):
        result = parts[vertexPropertyMap[name]].parseFloat
      else:
        result = default

    let x = getProp("x", 0.0)
    let y = getProp("y", 0.0)
    let z = getProp("z", 0.0)
    let nx = getProp("nx", 0.0)
    let ny = getProp("ny", 1.0)
    let nz = getProp("nz", 0.0)
    let u = getProp("s", 0.0)
    var v = getProp("t", 0.0)
    v = 1.0 - v
    let r = getProp("red", 255.0).uint8
    let g = getProp("green", 255.0).uint8
    let b = getProp("blue", 255.0).uint8
    let a = getProp("alpha", 255.0).uint8
    let color = packColor(r, g, b, a)

    out_vertices[i - bodyStartIndex] = Vertex(
      x: x.float32, y: y.float32, z: z.float32,
      xN: nx.float32, yN: ny.float32, zN: nz.float32,
      color: color,
      u: u.float32, v: v.float32
    )

  let faceLinesEnd = vertexLinesEnd + faceCount
  for i in vertexLinesEnd ..< faceLinesEnd:
    let parts = fileLines[i].split()
    let numVertsInFace = parts[0].parseInt
    case numVertsInFace
    of 3:
      let i0 = parts[1].parseInt.uint16
      let i1 = parts[2].parseInt.uint16
      let i2 = parts[3].parseInt.uint16
      out_indices.add([i0, i2, i1])
    of 4:
      let i0 = parts[1].parseInt.uint16
      let i1 = parts[2].parseInt.uint16
      let i2 = parts[3].parseInt.uint16
      let i3 = parts[4].parseInt.uint16
      out_indices.add([i0, i1, i2])
      out_indices.add([i0, i2, i3])
    else: discard

  echo "loadPly, Loaded PLY: $1 vertices, $2 indices" % [$out_vertices.len, $out_indices.len]
  return (out_vertices, out_indices)

proc saveMeshToCache*(fs: var RuntimeFS, path: string, vertices: seq[Vertex], indices: seq[uint16]) =
  echo "Saving baked mesh to cache: ", path
  var stream = newStringStream()
  try:
    stream.write(vertices.len.int32)
    stream.write(indices.len.int32)
    stream.writeData(vertices[0].addr, vertices.len * sizeof(Vertex))
    stream.writeData(indices[0].addr, indices.len * sizeof(uint16))
    fs.write(path, cast[seq[byte]](stream.data))
  finally:
    stream.close()

proc loadMeshFromCache*(fs: RuntimeFS, path: string): (seq[Vertex], seq[uint16]) =
  echo "Loading baked mesh from cache: ", path
  let contentOpt = fs.get(path)
  if contentOpt.isNone:
    return
  var stream = newStringStream(contentOpt.get())
  var vertices: seq[Vertex]
  var indices: seq[uint16]
  try:
    var vertCount, idxCount: int32
    stream.read(vertCount)
    stream.read(idxCount)
    vertices.setLen(vertCount.int)
    indices.setLen(idxCount.int)
    discard stream.readData(vertices[0].addr, vertCount * sizeof(Vertex))
    discard stream.readData(indices[0].addr, idxCount * sizeof(uint16))
  finally:
    stream.close()
  return (vertices, indices)

proc loadAndProcessMesh*(fs: var RuntimeFS, modelFilename: string, aoParams: AOBakeParams, texture: Image, sampler: Sampler): (Mesh, seq[Vertex], seq[uint16]) =
  var
    cpuVertices: seq[Vertex]
    cpuIndices: seq[uint16]

  let cacheFilename = modelFilename & ".baked_ao.bin"

  if fs.fileExists(cacheFilename):
    (cpuVertices, cpuIndices) = loadMeshFromCache(fs, cacheFilename)
  else:
    let modelContentOpt = fs.get(modelFilename)
    if modelContentOpt.isNone:
      return

    let fileLines = modelContentOpt.get().splitLines()
    let fileExt = modelFilename.splitFile.ext
    case fileExt.toLower()
    of ".ply":
      (cpuVertices, cpuIndices) = loadPly(fileLines)
    of ".obj":
      (cpuVertices, cpuIndices) = loadObj(fileLines)
    else:
      return

    if cpuVertices.len > 0:
      bakeBentNormalWithGrid(cpuVertices, cpuIndices, aoParams, gridResolution = 64)
      saveMeshToCache(fs, cacheFilename, cpuVertices, cpuIndices)

  if cpuVertices.len > 0 and cpuIndices.len > 0:
    let vbuf = sg.makeBuffer(BufferDesc(
      usage: BufferUsage(vertexBuffer: true),
      data: sg.Range(addr: cpuVertices[0].addr, size: cpuVertices.len * sizeof(Vertex))
    ))
    let ibuf = sg.makeBuffer(BufferDesc(
      usage: BufferUsage(indexBuffer: true),
      data: sg.Range(addr: cpuIndices[0].addr, size: cpuIndices.len * sizeof(uint16))
    ))
    result[0].indexCount = cpuIndices.len.int32
    result[0].bindings = Bindings(vertexBuffers: [vbuf], indexBuffer: ibuf)
    result[0].bindings.images[shd.imgUTexture] = texture
    result[0].bindings.samplers[shd.smpUSampler] = sampler
    result[1] = cpuVertices
    result[2] = cpuIndices

proc loadTexture*(fs: RuntimeFS, filename: string): sg.Image =
  let qoiContentOpt = fs.get(filename)
  if qoiContentOpt.isNone:
    return sg.Image()

  var qoiImage: QOIF
  try:
    let qoiContentStr = qoiContentOpt.get()
    qoiImage = decodeQOI(cast[seq[byte]](qoiContentStr))
  except Exception as e:
    return sg.Image()

  var finalPixelData: seq[byte]
  var finalPixelFormat: sg.PixelFormat
  if qoiImage.header.channels == qoi.RGBA:
    finalPixelData = qoiImage.data
    finalPixelFormat = sg.PixelFormat.pixelFormatRgba8
  else:
    finalPixelFormat = sg.PixelFormat.pixelFormatRgba8
    let numPixels = qoiImage.header.width.int * qoiImage.header.height.int
    finalPixelData = newSeq[byte](numPixels * 4)
    var srcIndex = 0
    var dstIndex = 0
    for i in 0 ..< numPixels:
      finalPixelData[dstIndex]   = qoiImage.data[srcIndex]
      finalPixelData[dstIndex+1] = qoiImage.data[srcIndex+1]
      finalPixelData[dstIndex+2] = qoiImage.data[srcIndex+2]
      finalPixelData[dstIndex+3] = 255.byte
      srcIndex += 3
      dstIndex += 4

  result = sg.makeImage(sg.ImageDesc(
    width: qoiImage.header.width.int32,
    height: qoiImage.header.height.int32,
    pixelFormat: finalPixelFormat,
    data: ImageData(
      subimage: [ [ sg.Range(addr: finalPixelData[0].addr, size: qoiImage.header.width.int32 * qoiImage.header.height.int32 * 4) ] ]
    )
  ))
