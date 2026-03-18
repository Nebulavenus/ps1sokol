import sokol/debugtext as sdtx
import sokol/gfx as sg
import math/vec3
import math/mat4
import types
import std/strformat
import audio
import camera

proc drawUI*(state: var State, proj, view: Mat4, canvasW, canvasH: float32) =
  sdtx.canvas(canvasW, canvasH)
  let gridW = 40.0f
  let gridH = 30.0f

  if state.gameState == GameState.MainMenu:
    let menuX = 12.0f
    let menuY = 10.0f
    sdtx.pos(menuX, menuY - 4)
    sdtx.color3f(0.2, 0.8, 1.0)
    sdtx.puts("PS1 SOKOL RACER")
    
    let items = ["START GAME", "CONTROLS", "QUIT"]
    for i, item in items:
      sdtx.pos(menuX, menuY + i.float * 2.0)
      if i == state.menu.selectedItem:
        sdtx.color3f(1.0, 1.0, 1.0)
        sdtx.puts(&"> {item}")
      else:
        sdtx.color3f(0.5, 0.5, 0.5)
        sdtx.puts(&"  {item}")

  elif state.gameState == GameState.CarSelection:
    let uiX = 2.0f
    let uiY = 4.0f
    let currentCar = state.availableCars[state.selectedCarIdx]
    
    sdtx.pos(uiX, uiY)
    sdtx.color3f(0.2, 0.8, 1.0)
    sdtx.puts("SELECT YOUR VEHICLE")
    
    sdtx.pos(uiX, uiY + 3.0)
    sdtx.color3f(1.0, 1.0, 1.0)
    sdtx.puts(&"NAME: {currentCar.name}")
    
    let stats = [
      ("POWER", currentCar.engineForce / 40.0),
      ("HANDLING", currentCar.turnTorque / 150.0),
      ("MAX SPEED", currentCar.maxSpeed / 50.0),
      ("GRIP", currentCar.baseGrip)
    ]
    
    for i, stat in stats:
      sdtx.pos(uiX, uiY + 6.0 + i.float * 1.5)
      sdtx.color3f(0.7, 0.7, 0.7)
      sdtx.puts(stat[0].cstring)
      
      let barLen = 10
      let filled = int(stat[1] * barLen.float)
      var barStr = ""
      for b in 0..<barLen:
        if b < filled: barStr.add("|")
        else: barStr.add(".")
      
      sdtx.pos(uiX + 12.0, uiY + 6.0 + i.float * 1.5)
      sdtx.color3f(1.0, 0.6, 0.0)
      sdtx.puts(&"[{barStr}]")

    sdtx.pos(uiX, uiY + 15.0)
    sdtx.color3f(0.5, 0.5, 0.5)
    sdtx.puts("A/D - CHANGE CAR")
    sdtx.pos(uiX, uiY + 16.5)
    sdtx.puts("ENTER - SELECT AND SETUP RACE")

  elif state.gameState == GameState.RaceSetup:
    let uiX = 2.0f
    let uiY = 4.0f
    sdtx.pos(uiX, uiY)
    sdtx.color3f(0.2, 0.8, 1.0)
    sdtx.puts("RACE SETUP")

    let items = [
      &"MODE: {state.gameMode}",
      &"OPPONENTS: {state.aiCount}",
      &"DIFFICULTY: {state.aiDifficulty}",
      "START RACE"
    ]

    for i, item in items:
      sdtx.pos(uiX, uiY + 4.0 + i.float * 2.0)
      if i == state.menu.selectedItem:
        sdtx.color3f(1.0, 1.0, 1.0)
        sdtx.puts(&"> {item}")
      else:
        sdtx.color3f(0.5, 0.5, 0.5)
        sdtx.puts(&"  {item}")

    sdtx.pos(uiX, uiY + 15.0)
    sdtx.color3f(0.5, 0.5, 0.5)
    sdtx.puts("A/D - ADJUST VALUES")
    sdtx.pos(uiX, uiY + 16.5)
    sdtx.puts("ESC - BACK TO CAR SELECTION")

  elif state.gameState == GameState.Paused:
    let menuX = 14.0f
    let menuY = 10.0f
    sdtx.pos(menuX, menuY - 2)
    sdtx.color3f(1.0, 1.0, 0.0)
    sdtx.puts("=== PAUSED ===")
    
    let items = ["RESUME", "RESTART", "MAIN MENU", "MUSIC VOL", "SFX VOL", "CONTROLS", "QUIT"]
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

  if state.gameState == GameState.Playing or state.gameState == GameState.Paused:
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

    if state.gameMode == GameMode.TofuDelivery:
      sdtx.pos(1, 7)
      sdtx.color3f(1.0, 1.0, 1.0)
      sdtx.puts(">> TOFU INTEGRITY")
      
      let integrity = state.tofuIntegrity
      let barLen = 20
      let filled = int(integrity * barLen.float)
      var barStr = ""
      for b in 0..<barLen:
        if b < filled: barStr.add("|")
        else: barStr.add(".")
      
      sdtx.pos(1, 8)
      if integrity > 0.5: sdtx.color3f(0.2, 1.0, 0.4)
      elif integrity > 0.2: sdtx.color3f(1.0, 1.0, 0.0)
      else: sdtx.color3f(1.0, 0.0, 0.0)
      sdtx.puts(&"[{barStr}] {integrity*100:3.0f}%")

      # NITRO HUD
      sdtx.pos(1, 10)
      sdtx.color3f(0.0, 0.8, 1.0)
      sdtx.puts(">> NITRO")
      let nitro = state.boostAmount
      let nBarLen = 15
      let nFilled = int(nitro * nBarLen.float)
      var nBarStr = ""
      for b in 0..<nBarLen:
        if b < nFilled: nBarStr.add("!")
        else: nBarStr.add(".")
      sdtx.pos(1, 11)
      if state.isBoosting: sdtx.color3f(1.0, 1.0, 1.0)
      else: sdtx.color3f(0.0, 0.6, 0.8)
      sdtx.puts(&"[{nBarStr}] {nitro*100:3.0f}%")

      if state.raceFinished:
        sdtx.pos(10, 12)
        sdtx.color3f(0.0, 1.0, 0.0)
        sdtx.puts("DELIVERY COMPLETE!")
      elif state.tofuIntegrity <= 0.0:
        sdtx.pos(10, 12)
        sdtx.color3f(1.0, 0.0, 0.0)
        sdtx.puts("DELIVERY FAILED - TOFU SMASHED!")

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

    let dashX = gridW - 16.0
    let dashY = gridH - 5.0
    sdtx.pos(dashX, dashY)
    sdtx.color3f(1.0, 0.2, 0.2)
    sdtx.puts(">> TELEMETRY")
    sdtx.color3f(1.0, 1.0, 1.0)
    sdtx.pos(dashX, dashY + 1)
    sdtx.puts(&"SPD {state.debugSpeed:5.1f} KM/H")
    sdtx.pos(dashX, dashY + 2)
    let rpm = state.debugRpm
    let rpmPct = clamp((rpm - 1000.0) / 5000.0, 0.0, 1.0)
    let rpmBars = int(rpmPct * 10)
    var rpmStr = ""
    for i in 0..<10:
      if i < rpmBars: rpmStr.add("|")
      else: rpmStr.add(".")
    if rpmPct > 0.8: sdtx.color3f(1.0, 0.0, 0.0)
    elif rpmPct > 0.5: sdtx.color3f(1.0, 1.0, 0.0)
    else: sdtx.color3f(0.0, 1.0, 0.0)
    sdtx.puts(&"RPM [{rpmStr}]")
    sdtx.color3f(1.0, 1.0, 1.0)
    sdtx.pos(dashX, dashY + 3)
    sdtx.puts(&"GEAR {state.debugGear}")

    for ai in state.aiCars:
      let (screenPos, visible) = worldToScreen(ai.position + vec3(0, 2.5, 0), proj, view, canvasW, canvasH)
      if visible:
        sdtx.pos(screenPos.x / 8.0, screenPos.y / 8.0)
        case ai.difficulty:
        of Difficulty.Easy: sdtx.color3f(0.5, 1.0, 0.5)
        of Difficulty.Medium: sdtx.color3f(1.0, 1.0, 0.5)
        of Difficulty.Hard: sdtx.color3f(1.0, 0.5, 0.5)
        sdtx.puts(&"{ai.name}")
        sdtx.pos(screenPos.x / 8.0, (screenPos.y / 8.0) + 1.0)
        sdtx.puts(&"[{ai.difficulty}]")

  sdtx.draw()
