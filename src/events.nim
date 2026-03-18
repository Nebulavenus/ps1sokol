import sokol/app as sapp
import math
import math/vec3
import math/mat4
import types
import audio
import level

when defined(emscripten):
  proc emscripten_run_script(script: cstring) {.importc, header: "<emscripten/emscripten.h>".}

proc event*(e: ptr sapp.Event, state: var State) =
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
        state.menu.itemCount = 7 # RESUME, RESTART, MAIN MENU, MUSIC, SFX, CONTROLS, QUIT
      elif state.gameState == GameState.Paused:
        state.gameState = GameState.Playing
      elif state.gameState == GameState.CarSelection:
        state.gameState = GameState.MainMenu
        state.menu.selectedItem = 0
        state.menu.itemCount = 3 # START, CONTROLS, QUIT
      elif state.gameState == GameState.RaceSetup:
        state.gameState = GameState.CarSelection
        state.menu.selectedItem = 0
        state.menu.itemCount = 0 # Not used for car selection grid but mod handles it
      elif state.gameState == GameState.ControlsMenu:
        state.gameState = state.previousGameState
    of keyCodeW, keyCodeUp:
      if state.gameState == GameState.Paused or state.gameState == GameState.MainMenu or state.gameState == GameState.RaceSetup:
        state.menu.selectedItem = (state.menu.selectedItem + state.menu.itemCount - 1) mod state.menu.itemCount
    of keyCodeS, keyCodeDown:
      if state.gameState == GameState.Paused or state.gameState == GameState.MainMenu or state.gameState == GameState.RaceSetup:
        state.menu.selectedItem = (state.menu.selectedItem + 1) mod state.menu.itemCount
    of keyCodeA, keyCodeLeft:
      if state.gameState == GameState.Paused:
        if state.menu.selectedItem == 3: # MUSIC VOLUME
          setMusicVolume(getMusicVolume() - 0.1)
        elif state.menu.selectedItem == 4: # SFX VOLUME
          setSfxVolume(getSfxVolume() - 0.1)
      elif state.gameState == GameState.CarSelection:
        state.selectedCarIdx = (state.selectedCarIdx + state.availableCars.len - 1) mod state.availableCars.len
      elif state.gameState == GameState.RaceSetup:
        if state.menu.selectedItem == 0: # MODE
          state.gameMode = if state.gameMode == GameMode.StandardRace: GameMode.TofuDelivery else: GameMode.StandardRace
        elif state.menu.selectedItem == 1: # OPPONENTS
          state.aiCount = max(0, state.aiCount - 1)
        elif state.menu.selectedItem == 2: # DIFFICULTY
          if state.aiDifficulty == Difficulty.Medium: state.aiDifficulty = Difficulty.Easy
          elif state.aiDifficulty == Difficulty.Hard: state.aiDifficulty = Difficulty.Medium
    of keyCodeD, keyCodeRight:
      if state.gameState == GameState.Paused:
        if state.menu.selectedItem == 3: # MUSIC VOLUME
          setMusicVolume(getMusicVolume() + 0.1)
        elif state.menu.selectedItem == 4: # SFX VOLUME
          setSfxVolume(getSfxVolume() + 0.1)
      elif state.gameState == GameState.CarSelection:
        state.selectedCarIdx = (state.selectedCarIdx + 1) mod state.availableCars.len
      elif state.gameState == GameState.RaceSetup:
        if state.menu.selectedItem == 0: # MODE
          state.gameMode = if state.gameMode == GameMode.StandardRace: GameMode.TofuDelivery else: GameMode.StandardRace
        elif state.menu.selectedItem == 1: # OPPONENTS
          state.aiCount = min(10, state.aiCount + 1)
        elif state.menu.selectedItem == 2: # DIFFICULTY
          if state.aiDifficulty == Difficulty.Easy: state.aiDifficulty = Difficulty.Medium
          elif state.aiDifficulty == Difficulty.Medium: state.aiDifficulty = Difficulty.Hard
    of keyCodeEnter, keyCodeSpace:
      if state.gameState == GameState.MainMenu:
        case state.menu.selectedItem
        of 0: state.gameState = GameState.CarSelection # GO TO CAR SELECTION
        of 1:
          state.previousGameState = state.gameState
          state.gameState = GameState.ControlsMenu # CONTROLS
        of 2: sapp.requestQuit() # QUIT
        else: discard
      elif state.gameState == GameState.CarSelection:
        state.gameState = GameState.RaceSetup
        state.menu.selectedItem = 0
        state.menu.itemCount = 4 # MODE, OPPONENTS, DIFFICULTY, START
      elif state.gameState == GameState.RaceSetup:
        if state.menu.selectedItem == 3: # START RACE
          state.gameState = GameState.Playing
          restartLevel(state)
      elif state.gameState == GameState.Paused:
        case state.menu.selectedItem
        of 0: state.gameState = GameState.Playing # RESUME
        of 1: # RESTART
          restartLevel(state)
          state.gameState = GameState.Playing
        of 2: # MAIN MENU
          state.gameState = GameState.MainMenu
          state.menu.selectedItem = 0
          state.menu.itemCount = 3 # START, CONTROLS, QUIT
          restartLevel(state)
        of 3, 4: discard # VOLUME (handled by left/right)
        of 5: # CONTROLS
          state.previousGameState = state.gameState
          state.gameState = GameState.ControlsMenu
        of 6: sapp.requestQuit() # QUIT
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
      of keyCodeLeftShift: state.input.nitroPressed = isDown
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
