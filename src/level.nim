import os
import math
import math/vec3
import math/mat4
import types
import rtfs
import mesh_loader
import sokol/gfx as sg
import std/strformat
import std/random
import physics
import aobaker

proc extractPathFromRoadMesh*(vertices: seq[Vertex], indices: seq[uint16]): seq[Vec3] =
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
  
  # Heuristic: Find the spine of the road by jumping along it and averaging nearby triangles
  for _ in 0 ..< 500:
    var closestIdx = -1
    var closestDist = 1e9
    
    for i in 0 ..< rawNodes.len:
      if visited[i]: continue
      # Height-aware distance: penalize vertical distance to stay on the current road level
      let dx = rawNodes[i].x - currentPos.x
      let dy = rawNodes[i].y - currentPos.y
      let dz = rawNodes[i].z - currentPos.z
      let dWeighted = sqrt(dx*dx + dz*dz + dy*dy * 25.0) 
      if dWeighted < closestDist and dWeighted > 5.0:
        closestDist = dWeighted
        closestIdx = i
    
    if closestIdx != -1:
      var sum = vec3(0,0,0)
      var count = 0
      let anchor = rawNodes[closestIdx]
      for i in 0 ..< rawNodes.len:
        if visited[i]: continue
        # Only average triangles that are horizontally close AND at a similar height
        let dx = rawNodes[i].x - anchor.x
        let dz = rawNodes[i].z - anchor.z
        let dy = rawNodes[i].y - anchor.y
        let distHorizSq = dx*dx + dz*dz
        if distHorizSq < 20.0*20.0 and abs(dy) < 5.0:
          sum = sum + rawNodes[i]
          count += 1
          visited[i] = true
      
      if count > 0:
        let spineNode = sum / count.float
        sortedNodes.add(spineNode)
        currentPos = spineNode
    else:
      break

  if sortedNodes.len > 4:
    var curvedNodes: seq[Vec3]
    for i in 0 ..< sortedNodes.len:
      let p0 = sortedNodes[(i + sortedNodes.len - 1) mod sortedNodes.len]
      let p1 = sortedNodes[i]
      let p2 = sortedNodes[(i + 1) mod sortedNodes.len]
      let p3 = sortedNodes[(i + 2) mod sortedNodes.len]
      
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
    return curvedNodes

  return sortedNodes

const BOT_NAMES = [
  "TAKUMI", "KEISUKE", "RYOSUKE", "BUNTA", "IKETANI",
  "KENJI", "SHINGO", "MAKO", "SAYUKI", "KENTA",
  "SUDO", "SEIJI", "SAKAI", "KAWAI", "TOMO"
]

proc initBots*(state: var State) =
  state.aiCars = @[]
  if state.pathNodes.len > 10:
    for i in 0 ..< state.aiCount:
      var ai: AIVehicle
      # Start bots slightly behind or ahead of player spawn
      let startIdx = (i + 1) * 4
      if startIdx >= state.pathNodes.len: break
      
      ai.position = state.pathNodes[startIdx] + vec3(0, 1, 0)
      ai.targetNode = (startIdx + 1) mod state.pathNodes.len
      let toTarget = norm(state.pathNodes[ai.targetNode] - ai.position)
      ai.yaw = (arctan2(toTarget.x, toTarget.z) + PI) * (180.0 / PI)
      ai.rotation = rotate(ai.yaw, vec3(0, 1, 0))
      
      # Vary difficulty around global setting
      let variation = rand(-1..1)
      var botDiff = state.aiDifficulty
      if variation == -1:
        if botDiff == Difficulty.Medium: botDiff = Difficulty.Easy
        elif botDiff == Difficulty.Hard: botDiff = Difficulty.Medium
      elif variation == 1:
        if botDiff == Difficulty.Easy: botDiff = Difficulty.Medium
        elif botDiff == Difficulty.Medium: botDiff = Difficulty.Hard
      
      ai.difficulty = botDiff
      ai.speedMultiplier = 0.9f + rand(0.2f) # 0.9 to 1.1 multiplier
      ai.name = if i < BOT_NAMES.len: BOT_NAMES[i] else: &"BOT {i+1}"
      state.aiCars.add(ai)
  echo &"Spawned {state.aiCars.len} bots"

proc restartLevel*(state: var State) =
  if state.gameState == GameState.MainMenu:
    state.player.position = vec3(0.0, 12, 25.0)
    state.player.yaw = 0.0
  elif state.pathNodes.len > 1:
    state.player.position = state.pathNodes[0] + vec3(0, 1, 0)
    let toNext = norm(state.pathNodes[1] - state.pathNodes[0])
    state.player.yaw = (arctan2(toNext.x, toNext.z) + PI) * (180.0 / PI)
  else:
    state.player.position = vec3(0.0, 12, 25.0)
    state.player.yaw = 0.0

  state.player.velocity = vec3(0, 0, 0)
  state.player.angularVelocity = 0.0
  state.player.rotation = rotate(state.player.yaw, vec3(0, 1, 0))
  state.currentCheckpointIdx = 0
  state.lapCount = 0
  state.lapStartTime = state.time
  state.lastLapTime = 0.0
  state.replayBuffer = @[]
  state.isReplaying = false

  if state.gameState == GameState.Playing:
    initBots(state)

  echo "Level Restarted"

proc loadLevel*(state: var State, fs: var RuntimeFS, mapDir: string) =
  echo &"Loading level: {mapDir}"
  
  let aoParams = AOBakeParams(numRays: 64, maxDistance: 2.0, intensity: 1.0, bias: 0.001)
  let pointSmp = sg.makeSampler(sg.SamplerDesc(minFilter: filterNearest, magFilter: filterNearest))
  
  let trackTexture1 = loadTexture(state, fs, mapDir/"track_road.qoi")
  let trackTexture2 = loadTexture(state, fs, mapDir/"track_shape.qoi")
  let trackTexture3 = loadTexture(state, fs, mapDir/"track_trees.qoi")
  
  state.trackMesh1 = loadAndProcessMesh(state, fs, mapDir/"track_road.ply", aoParams, trackTexture1, pointSmp)
  state.trackMesh2 = loadAndProcessMesh(state, fs, mapDir/"track_shape.ply", aoParams, trackTexture2, pointSmp)
  state.trackMesh3 = loadAndProcessMesh(state, fs, mapDir/"track_trees.ply", aoParams, trackTexture3, pointSmp)
  state.trackMesh4 = loadAndProcessMesh(state, fs, mapDir/"track_barrier.ply", aoParams, trackTexture1, pointSmp)
  state.trackMesh5 = loadAndProcessMesh(state, fs, mapDir/"track_borders.ply", aoParams, trackTexture1, pointSmp)
  
  (state.roadCollisionVertices, state.roadCollisionIndices) = loadAndProcessMeshCollision(fs, mapDir/"track_road.ply")
  state.roadGrid = initUniformGrid(state.roadCollisionVertices, 64)
  populateGrid(state.roadGrid, state.roadCollisionVertices, state.roadCollisionIndices)
  
  (state.barrierCollisionVertices, state.barrierCollisionIndices) = loadAndProcessMeshCollision(fs, mapDir/"track_barrier.ply")
  state.barrierGrid = initUniformGrid(state.barrierCollisionVertices, 64)
  populateGrid(state.barrierGrid, state.barrierCollisionVertices, state.barrierCollisionIndices)
  
  state.pathNodes = extractPathFromRoadMesh(state.roadCollisionVertices, state.roadCollisionIndices)
  state.checkpoints = @[]
  for i in countup(0, state.pathNodes.len - 1, 4):
    state.checkpoints.add(Checkpoint(pos: state.pathNodes[i], radius: 18.0))

  restartLevel(state)
