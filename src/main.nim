import sokol/log as slog
import sokol/app as sapp
import sokol/gfx as sg
import sokol/glue as sglue
import sokol/debugtext as sdtx
import shaders/default as shd
import math/vec2
import math/vec3
import math/mat4
import strutils
import os
import std/strformat
import audio
import rtfs
import types
import mesh_loader
import aobaker
import physics
import options
import culling
import tables

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

  let aoParams = AOBakeParams(
    numRays: 64,
    maxDistance: 2.0,
    intensity: 1.0,
    bias: 0.001,
  )
  state.aoShadowStrength = 1.0
  state.skyLightColor = vec3(0.4, 0.5, 0.8)
  state.skyLightIntensity = 0.0
  state.groundLightColor = vec3(0.6, 0.4, 0.3)
  state.groundLightIntensity = 0.0

  let pointSmp = sg.makeSampler(sg.SamplerDesc(
    minFilter: filterNearest,
    magFilter: filterNearest,
  ));

  loadMusic(ASSETS_FS, "music/sunset_relay.qoa")

  # Load the meshes. One function handles everything
  let trackTexture1 = loadTexture(state, ASSETS_FS, "map2"/"track_road.qoi")
  let trackTexture2 = loadTexture(state, ASSETS_FS, "map2"/"track_shape.qoi")
  let trackTexture3 = loadTexture(state, ASSETS_FS, "map2"/"track_trees.qoi")
  let carTexture = loadTexture(state, ASSETS_FS, "car"/"trueno.qoi")

  state.trackMesh1 = loadAndProcessMesh(state, ASSETS_FS, "map2"/"track_road.ply", aoParams, trackTexture1, pointSmp)
  state.trackMesh2 = loadAndProcessMesh(state, ASSETS_FS, "map2"/"track_shape.ply", aoParams, trackTexture2, pointSmp)
  state.trackMesh3 = loadAndProcessMesh(state, ASSETS_FS, "map2"/"track_trees.ply", aoParams, trackTexture3, pointSmp)
  let barrierMesh = loadAndProcessMesh(state, ASSETS_FS, "map2"/"track_barrier.ply", aoParams, trackTexture1, pointSmp)
  state.carMesh1 = loadAndProcessMesh(state, ASSETS_FS, "car"/"trueno.ply", aoParams, carTexture, pointSmp)
  state.carMesh2 = loadAndProcessMesh(state, ASSETS_FS, "car"/"trueno_back.ply", aoParams, carTexture, pointSmp)
  state.carMesh3 = loadAndProcessMesh(state, ASSETS_FS, "car"/"trueno_front.ply", aoParams, carTexture, pointSmp)

  (state.roadCollisionVertices, state.roadCollisionIndices) = loadAndProcessMeshCollision(ASSETS_FS, "map2"/"track_road.ply")
  state.roadGrid = initUniformGrid(state.roadCollisionVertices, 64)
  populateGrid(state.roadGrid, state.roadCollisionVertices, state.roadCollisionIndices)

  (state.barrierCollisionVertices, state.barrierCollisionIndices) = loadAndProcessMeshCollision(ASSETS_FS, "map2"/"track_barrier.ply")
  state.barrierGrid = initUniformGrid(state.barrierCollisionVertices, 64)
  populateGrid(state.barrierGrid, state.barrierCollisionVertices, state.barrierCollisionIndices)

  state.player.position = vec3(0.0, 12, 25.0)
  state.player.velocity = vec3(0, 0, 0)
  state.player.yaw = 0.0
  state.player.angularVelocity = 0.0
  state.player.rotation = rotate(state.player.yaw, vec3(0, 1, 0))

  state.cameraPos = vec3(0.0, 10.0, 2.0)
  state.cameraOffsetY = 5.0
  state.cameraTarget = state.player.position

proc frame() {.cdecl.} =
  let dt = sapp.frameDuration()

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
    state.player.position = nextPosition

    var surfaceUp = vec3.up()
    let surfaceInfoOpt = getSurfaceInfo(state, state.player.position)

    if surfaceInfoOpt.isSome:
      let surfaceInfo = surfaceInfoOpt.get()
      state.player.position.y = surfaceInfo.pos.y + 0.9
      surfaceUp = surfaceInfo.normal
    else:
      state.player.position.y -= 9.8 * dt

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

  updateCamera(state, dt)
  audioGenerateSamples()

  let fsParams = computeFsParams()
  let proj = persp(60.0f, sapp.widthf() / sapp.heightf(), 0.01f, 150.0f)
  let view = lookat(state.cameraPos, state.cameraTarget, vec3.up())

  sg.beginPass(Pass(action: passAction, swapchain: sglue.swapchain()))
  sg.applyPipeline(state.pip)
  sg.applyUniforms(shd.ubFsParams, sg.Range(addr: fsParams.addr, size: fsParams.sizeof))

  let trackModel = identity()
  var trackVsParams = shd.VsParams(
    u_mvp: proj * view * trackModel,
    u_model: trackModel,
    u_camPos: state.cameraPos,
    u_jitterAmount: 240.0,
  )
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

  let carModel = translate(state.player.position) * state.player.rotation
  var carVsParams = shd.VsParams(
    u_mvp: proj * view * carModel,
    u_model: carModel,
    u_camPos: state.cameraPos,
    u_jitterAmount: sapp.heightf(),
  )
  sg.applyBindings(state.carMesh1.bindings)
  sg.applyUniforms(shd.ubVsParams, sg.Range(addr: carVsParams.addr, size: carVsParams.sizeof))
  sg.draw(0, state.carMesh1.indexCount, 1)
  sg.applyBindings(state.carMesh2.bindings)
  sg.applyUniforms(shd.ubVsParams, sg.Range(addr: carVsParams.addr, size: carVsParams.sizeof))
  sg.draw(0, state.carMesh2.indexCount, 1)
  sg.applyBindings(state.carMesh3.bindings)
  sg.applyUniforms(shd.ubVsParams, sg.Range(addr: carVsParams.addr, size: carVsParams.sizeof))
  sg.draw(0, state.carMesh3.indexCount, 1)

  sdtx.canvas(sapp.widthf()*0.5, sapp.heightf()*0.5)
  sdtx.origin(1.0, 1.0)
  sdtx.home()
  sdtx.color3f(1.0, 1.0, 0.0)
  sdtx.puts((&"Speed: {state.debugSpeed:5.2f}\n").cstring)
  sdtx.puts((&"RPM: {state.debugRpm.int32}\n").cstring)
  sdtx.puts((&"Gear: {state.debugGear}\n").cstring)
  sdtx.draw()

  sg.endPass()
  sg.commit()

proc cleanup() {.cdecl.} =
  clearResources(state)
  sdtx.shutdown()
  audioShutdown()
  sg.shutdown()

proc event(e: ptr sapp.Event) {.cdecl.} =
  if e.`type` == EventType.eventTypeFocused:
    state.gameHasFocus = true
  elif e.`type` == EventType.eventTypeUnfocused:
    state.gameHasFocus = false

  if e.`type` == EventType.eventTypeMouseDown:
    if not state.gameHasFocus:
      state.gameHasFocus = true
      when defined(emscripten):
        emscripten_run_script("document.getElementById('canvas').focus();")

  if e.`type` == EventType.eventTypeMouseScroll:
    state.cameraOffsetY += e.scrollY * 0.5
    state.cameraOffsetY = max(state.cameraOffsetY, 0.0)

  if e.`type` == EventType.eventTypeKeyDown or e.`type` == EventType.eventTypeKeyUp:
    let step: float32 = 0.05
    let isDown = e.`type` == EventType.eventTypeKeyDown
    case e.keyCode
    of keyCodeEscape: sapp.requestQuit()
    of keyCodeW: state.input.accelerate = isDown
    of keyCodeS: state.input.brake = isDown
    of keyCodeA: state.input.turnLeft = isDown
    of keyCodeD: state.input.turnRight = isDown
    of keyCodeSpace: state.input.drift = isDown
    of keyCodeR:
      state.player.position = vec3(0.0, 12, 25.0)
      state.player.yaw = 0
    of keyCode1: state.aoShadowStrength = max(0.0, state.aoShadowStrength - step)
    of keyCode2: state.aoShadowStrength += step
    of keyCode3: state.skyLightIntensity = max(0.0, state.skyLightIntensity - step)
    of keyCode4: state.skyLightIntensity += step
    of keyCode5: state.groundLightIntensity = max(0.0, state.groundLightIntensity - step)
    of keyCode6: state.groundLightIntensity += step
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
