import sokol/log as slog
import sokol/app as sapp
import sokol/gfx as sg
import sokol/glue as sglue
import sokol/debugtext as sdtx
import shaders/default as shd
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
import particles
import ai
import camera
import level
import renderer
import ui
import events

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
  initSpritePipeline(state)
  initShadowTexture(state)
  initParticleTexture(state)
  initQuadBuffers(state)
  initOffscreen(state)
  initPostfxPipeline(state)
  initScreenQuad(state)
  randomize()
  
  state.aoShadowStrength = 1.0
  state.skyLightColor = vec3(0.4, 0.5, 0.8)
  state.skyLightIntensity = 0.0
  state.groundLightColor = vec3(0.6, 0.4, 0.3)
  state.groundLightIntensity = 0.0
  
  loadMusicPlaylist(ASSETS_FS, "music")
  
  state.availableCars = @[
    CarConfig(name: "TRUENO GT", modelPath: "car"/"trueno.ply", texturePath: "car"/"trueno.qoi", engineForce: 28.0, brakeForce: 25.0, maxSpeed: 32.0, baseGrip: 0.96, turnTorque: 90.0, baseIdlePitch: 0.0, pitchMultiplier: 1.0, maxRpm: 7500.0, gears: 5),
    CarConfig(name: "SILVIA S13", modelPath: "car"/"silvia.ply", texturePath: "car"/"silvia.qoi", engineForce: 35.0, brakeForce: 30.0, maxSpeed: 40.0, baseGrip: 0.98, turnTorque: 110.0, baseIdlePitch: 5.0, pitchMultiplier: 1.2, maxRpm: 9000.0, gears: 6),
    CarConfig(name: "LADA DRIFT", modelPath: "car"/"lada.ply", texturePath: "car"/"lada.qoi", engineForce: 30.0, brakeForce: 22.0, maxSpeed: 30.0, baseGrip: 0.85, turnTorque: 130.0, baseIdlePitch: -5.0, pitchMultiplier: 0.9, maxRpm: 6500.0, gears: 5)
  ]
  state.selectedCarIdx = 0

  let aoParams = AOBakeParams(numRays: 64, maxDistance: 2.0, intensity: 1.0, bias: 0.001)
  let pointSmp = sg.makeSampler(sg.SamplerDesc(minFilter: filterNearest, magFilter: filterNearest))
  
  let truenoTex = loadTexture(state, ASSETS_FS, "car"/"trueno.qoi")
  var truenoMeshes: seq[Mesh]
  truenoMeshes.add loadAndProcessMesh(state, ASSETS_FS, "car"/"trueno.ply", aoParams, truenoTex, pointSmp)
  state.carMeshes.add(truenoMeshes)

  let silviaTex = loadTexture(state, ASSETS_FS, "car"/"silvia.qoi")
  var silviaMeshes: seq[Mesh]
  silviaMeshes.add loadAndProcessMesh(state, ASSETS_FS, "car"/"silvia.ply", aoParams, silviaTex, pointSmp)
  state.carMeshes.add(silviaMeshes)

  let ladaTex = loadTexture(state, ASSETS_FS, "car"/"lada.qoi")
  var ladaMeshes: seq[Mesh]
  ladaMeshes.add loadAndProcessMesh(state, ASSETS_FS, "car"/"lada.ply", aoParams, ladaTex, pointSmp)
  state.carMeshes.add(ladaMeshes)
  
  state.cameraTarget = state.player.position
  state.lastEmitPos = state.player.position

  state.aiCount = 3
  state.aiDifficulty = Difficulty.Medium
  state.gameMode = GameMode.StandardRace
  state.tofuIntegrity = 1.0
  state.raceFinished = false

  loadLevel(state, ASSETS_FS, "map2")

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

  let currentCar = if state.availableCars.len > 0: state.availableCars[state.selectedCarIdx] else: CarConfig()

  if state.gameState == GameState.Playing:
    if not state.raceFinished and state.tofuIntegrity > 0.0:
      updateAI(state, dt)
    
    block VehiclePhysics:
      let engineForce = currentCar.engineForce
      let brakeForce = currentCar.brakeForce
      const drag = 0.8
      const angularDrag = 1.1
      let baseGrip = currentCar.baseGrip
      const driftGripMultiplier = 0.3
      const driftTurningMultiplier = 2.2
      let lowSpeedTurnTorque = currentCar.turnTorque
      let highSpeedTurnTorque = currentCar.turnTorque * 0.4
      const speedForMaxTurnDampening = 40.0
      let prevVelocity = state.player.velocity
      let forwardDir = state.player.rotation * vec3(0, 0, -1)
      
      let canMove = not state.raceFinished and state.tofuIntegrity > 0.0
      
      if state.input.accelerate and canMove:
        state.player.velocity += forwardDir * engineForce * dt
      if state.input.brake and canMove:
        if len(state.player.velocity) > 5.0:
          state.player.velocity -= norm(state.player.velocity) * brakeForce * dt
        else:
          state.player.velocity -= forwardDir * engineForce * dt
      
      let currentSpd = len(state.player.velocity)
      if currentSpd > currentCar.maxSpeed:
        state.player.velocity = norm(state.player.velocity) * currentCar.maxSpeed

      let currentSpeed1 = len(state.player.velocity)
      let turnDampeningFactor = clamp(currentSpeed1 / speedForMaxTurnDampening, 0.0, 1.0)
      var effectiveTurningTorque = lerp(lowSpeedTurnTorque, highSpeedTurnTorque, turnDampeningFactor)
      if state.input.drift and canMove:
        effectiveTurningTorque *= driftTurningMultiplier
      if state.input.turnLeft and canMove:
        state.player.angularVelocity += effectiveTurningTorque * dt
      if state.input.turnRight and canMove:
        state.player.angularVelocity -= effectiveTurningTorque * dt

      # Nitro Logic
      state.isBoosting = state.input.nitroPressed and state.boostAmount > 0.0 and canMove
      if state.isBoosting:
        state.player.velocity += forwardDir * (engineForce * 1.5) * dt
        state.boostAmount -= dt * 0.3 # Consume nitro
        if state.boostAmount < 0: state.boostAmount = 0
      else:
        state.boostAmount = min(1.0, state.boostAmount + dt * 0.05) # Slow recharge

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
        let wallNormal = norm(collisionInfo.pushOut)
        let velAlongNormal = dot(state.player.velocity, wallNormal)
        if velAlongNormal < 0:
          state.player.velocity = state.player.velocity * 0.95
          state.player.velocity -= wallNormal * velAlongNormal * 1.05
          
          # Tofu penalty
          if state.gameMode == GameMode.TofuDelivery and not state.raceFinished:
            let impactForce = abs(velAlongNormal)
            if impactForce > 5.0:
              state.tofuIntegrity = max(0.0, state.tofuIntegrity - 0.1)
              if state.tofuIntegrity <= 0.0:
                # Failure logic could go here, but for now we just show UI
                discard

      state.player.position = nextPosition
      var surfaceUp = vec3.up()
      let surfaceInfoOpt = getSurfaceInfo(state, state.player.position)
      if surfaceInfoOpt.isSome:
        let surfaceInfo = surfaceInfoOpt.get()
        state.player.position.y = surfaceInfo.pos.y + 0.9
        surfaceUp = surfaceInfo.normal
      if state.player.position.y < -50.0 or surfaceInfoOpt.isNone:
        let lastCpIdx = (state.currentCheckpointIdx + state.checkpoints.len - 1) mod state.checkpoints.len
        let respawnPos = state.checkpoints[lastCpIdx].pos
        state.player.position = respawnPos + vec3(0, 2, 0)
        state.player.velocity = vec3(0, 0, 0)
        let toNext = norm(state.checkpoints[state.currentCheckpointIdx].pos - respawnPos)
        state.player.yaw = (arctan2(toNext.x, toNext.z) + PI) * (180.0 / PI)
        state.player.rotation = rotate(state.player.yaw, vec3(0, 1, 0))

      let currentSpeed = len(state.player.velocity)
      let carAccel = (currentSpeed - len(prevVelocity)) / dt
      updateEngineSound(currentCar, currentSpeed, carAccel, state.input.drift, state.debugSpeed, state.debugRpm, state.debugGear)
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
      state.player.rotation = fromCols(newRight, surfaceUp, -finalForward, vec3(0,0,0))
      
      if state.input.drift:
        let carRot = state.player.rotation
        let distMoved = len(state.player.position - state.lastEmitPos)
        let numSteps = clamp(int(distMoved / 0.05), 1, 15)
        for s in 0 ..< numSteps:
          let t = s.float / numSteps.float
          let lerpPos = vec3.lerpV(state.lastEmitPos, state.player.position, t)
          let leftEmitPos = lerpPos + (carRot * vec3(-0.8, -0.5, 1.2))
          let rightEmitPos = lerpPos + (carRot * vec3(0.8, -0.5, 1.2))
          let smokeVel = (carRot * vec3(0, 0.5, 1.0)) * 2.0
          let randVel = vec3(rand(-0.5..0.5), rand(0.0..0.5), rand(-0.5..0.5))
          emitParticle(state, leftEmitPos, smokeVel + randVel, packColor(200, 200, 200, 150), 1.0)
          emitParticle(state, rightEmitPos, smokeVel + randVel, packColor(200, 200, 200, 150), 1.0)
      
      let exhaustPos = state.player.position + (state.player.rotation * vec3(-0.6, -0.4, 1.5))
      let exhaustVel = (state.player.rotation * vec3(0, 0.2, 0.5)) + vec3(rand(-0.1..0.1), rand(0.1..0.2), rand(-0.1..0.1))
      emitParticle(state, exhaustPos, exhaustVel, packColor(150, 150, 150, 100), 0.5)
      state.lastEmitPos = state.player.position

    block GameplayLogic:
      if state.checkpoints.len > 0:
        let nextCp = state.checkpoints[state.currentCheckpointIdx]
        if len(state.player.position - nextCp.pos) < nextCp.radius:
          state.currentCheckpointIdx = (state.currentCheckpointIdx + 1) mod state.checkpoints.len
          if state.currentCheckpointIdx == 0:
            state.lapCount += 1
            state.lastLapTime = state.time - state.lapStartTime
            if state.bestLapTime == 0.0 or state.lastLapTime < state.bestLapTime:
              state.bestLapTime = state.lastLapTime
            state.lapStartTime = state.time
            
            if state.gameMode == GameMode.TofuDelivery:
              state.raceFinished = true

      # Nitro Pickup Logic
      var i = 0
      while i < state.nitroPowerups.len:
        if len(state.player.position - state.nitroPowerups[i]) < 3.0:
          state.boostAmount = min(1.0, state.boostAmount + 0.3)
          state.nitroPowerups.delete(i)
          # Spawn some "pickup" particles
          for _ in 0..10:
            let vel = vec3(rand(-2.0..2.0), rand(1.0..4.0), rand(-2.0..2.0))
            emitParticle(state, state.player.position, vel, packColor(0, 255, 255, 200), 0.8)
        else:
          i += 1

      if state.isReplaying:
        if state.replayBuffer.len > 0:
          let frame = state.replayBuffer[state.replayIndex]
          state.player.position = frame.pos
          state.player.yaw = frame.yaw
          state.player.rotation = rotate(state.player.yaw, vec3(0, 1, 0))
          state.replayIndex = (state.replayIndex + 1) mod state.replayBuffer.len
      else:
        state.replayBuffer.add(ReplayFrame(pos: state.player.position, yaw: state.player.yaw))
        if state.replayBuffer.len > 10000:
          state.replayBuffer.delete(0)

  updateParticles(state, dt)
  
  if state.gameState == GameState.Playing:
    updateCamera(state, dt)
  elif state.gameState == GameState.MainMenu or state.gameState == GameState.CarSelection:
    let rotateSpeed = 60.0f
    state.player.yaw += rotateSpeed * dt
    if state.player.yaw >= 360.0: state.player.yaw -= 360.0
    let pitch = sin(state.time * 2.0) * 2.0f
    let roll = cos(state.time * 1.5) * 3.0f
    state.player.rotation = rotate(state.player.yaw, vec3(0, 1, 0)) * rotate(pitch, vec3(1, 0, 0)) * rotate(roll, vec3(0, 0, 1))
    let radius = 12.0f
    let camSpeed = 0.3f
    state.cameraPos = state.player.position + vec3(cos(state.time * camSpeed) * radius, 4.0, sin(state.time * camSpeed) * radius)
    state.cameraTarget = state.player.position + vec3(0, 0.5, 0)
  
  audioGenerateSamples(state.gameState == GameState.Playing)
  let fsParams = computeFsParams(state)
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
  if isMeshVisible(state.trackMesh5, state.cameraPos, camForward):
    sg.applyBindings(state.trackMesh5.bindings)
    sg.applyUniforms(shd.ubVsParams, sg.Range(addr: trackVsParams.addr, size: trackVsParams.sizeof))
    sg.draw(0, state.trackMesh5.indexCount, 1)
 
  drawShadow(state, proj, view)
  drawCheckpoints(state, proj, view)
  drawPowerups(state, proj, view)
  drawParticles(state, proj, view)
  
  sg.applyPipeline(state.pip)
  for i, ai in state.aiCars:
    let aiModel = translate(ai.position) * ai.rotation
    let aiCarIdx = (i + 1) mod state.carMeshes.len
    drawVehicle(state, proj, view, aiModel, state.cameraPos, aiCarIdx)

  if state.cameraMode == CameraMode.Follow or state.gameState == GameState.MainMenu or state.gameState == GameState.CarSelection:
    let playerModel = translate(state.player.position) * state.player.rotation
    drawVehicle(state, proj, view, playerModel, state.cameraPos, state.selectedCarIdx)

  sg.endPass()
  sg.beginPass(Pass(action: passAction, swapchain: sglue.swapchain()))
  drawPostfx(state)
  drawUI(state, proj, view, 320.0f, 240.0f)
  sg.endPass()
  sg.commit()

proc cleanup() {.cdecl.} =
  clearResources(state)
  sdtx.shutdown()
  audioShutdown()
  sg.shutdown()

proc event_callback(e: ptr sapp.Event) {.cdecl.} =
  event(e, state)

sapp.run(sapp.Desc(
  initCb: init,
  frameCb: frame,
  eventCb: event_callback,
  cleanupCb: cleanup,
  windowTitle: "Game",
  width: 640,
  height: 480,
  sampleCount: 0,
  icon: IconDesc(sokol_default: true),
  logger: sapp.Logger(fn: slog.fn)
))
