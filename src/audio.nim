import sokol/app as sapp
import sokol/audio as saudio
import sokol/log as slog
import math
import math/vec3 # For len() and dot() if needed for more complex sound logic
import std/random # For noise generation
import strutils # For cstring
import streams
import strformat
import rtfs, qoa, os

const
  DEFAULT_MUSIC_VOLUME = 1.0 # Default music volume (0.0 to 1.0)
  SAUDIO_SAMPLE_RATE = 44100.0   # Standard sample rate
  SAUDIO_NUM_SAMPLES = 2048       # Number of samples to push per chunk (power of 2 is good)

  # --- Engine Sound Parameters ---
  MAX_HARMONICS = 3              # Number of harmonic oscillators
  # Multipliers for each harmonic (e.g., 1st, 2nd, 3rd multiples of fundamental)
  HARMONIC_FREQS = [1.0, 1.5, 2.5]

  # Base volume for each harmonic at low RPM (idle)
  # Tune these to shape the idle sound
  HARMONIC_LEVELS_LOW_RPM = [0.8, 0.3, 0.1]
  # Base volume for each harmonic at high RPM (revving)
  # Tune these to shape the high-rev sound
  HARMONIC_LEVELS_HIGH_RPM = [0.7, 0.4, 0.2]

  # --- Gear Shifting Parameters ---
  GAME_MAX_SPEED = 35.0          # Estimated max speed of the car for normalization
  NUM_GEARS = 5
  # Speed ranges for each gear. Gear 0 is unused.
  GEAR_MIN_SPEED = [0.0, 0.0, 5.0, 10.0, 18.0, 25.0]
  GEAR_MAX_SPEED = [0.0, 8.0, 15.0, 24.0, 30.0, 35.0]
  UPSHIFT_RPM = 5500.0           # RPM at which we shift up
  DOWNSHIFT_RPM = 2500.0         # RPM at which we shift down
  GEAR_SHIFT_COOLDOWN = 0.25     # Seconds between gear shifts

  # --- New Drift Sound Parameters ---
  DRIFT_RPM_BOOST = 600.0        # How much RPM boosts when drifting
  MAX_SCREECH_VOLUME = 0.6       # INCREASE THIS: Max volume of the tire screech (try 0.8 or 1.0)
  SCREECH_ATTACK_SPEED = 15.0    # INCREASE THIS: How fast screech volume ramps up (make it punchier)
  SCREECH_DECAY_SPEED = 8.0      # INCREASE THIS: How fast screech volume ramps down (so it lingers a bit)

  SCREECH_BASE_FREQ = 1200.0     # Base frequency of the screech sound (e.g., 1200 Hz)
  SCREECH_FREQ_JITTER = 200.0    # How much the screech frequency can randomly vary
  SCREECH_NOISE_MIX = 0.3        # How much white noise to mix with the high-freq tone (0.0 to 1.0)

type
  AudioState = object
    oscillatorPhases: array[MAX_HARMONICS, float32] # Each harmonic needs its own phase
    screechPhase: float32

    # --- Engine & Vehicle State ---
    targetRpm: float32
    currentRpm: float32
    engineFrequency: float32
    engineVolume: float32
    harmonicsVolumes: array[MAX_HARMONICS, float32]
    engineNoiseVolume: float32
    currentScreechVolume: float32
    lastOutputSample: float32

    # --- Gear Shifting State ---
    currentGear: int
    gearShiftTimer: float32

    # Music playback state
    musicPcm: seq[int16]
    musicPosition: int
    musicPlaying: bool
    musicVolume: float32
    musicChannels: int

var audioState: AudioState
var audioSamples: array[SAUDIO_NUM_SAMPLES, float32] # Buffer for samples

proc lerp*(a, b: float32, t: float32): float32 {.inline.} =
  return a + (b - a) * t

proc audioInit*() =
  saudio.setup(saudio.Desc(
    logger: saudio.Logger(fn: slog.fn),
    sampleRate: SAUDIO_SAMPLE_RATE.int32
  ))
  for i in 0..<MAX_HARMONICS:
    audioState.oscillatorPhases[i] = 0.0
  audioState.targetRpm = 0.0
  audioState.currentRpm = 0.0
  audioState.engineFrequency = 0.0
  audioState.engineVolume = 0.0
  audioState.engineNoiseVolume = 0.0
  audioState.currentScreechVolume = 0.0
  audioState.screechPhase = 0.0 # Initialize screech phase
  audioState.lastOutputSample = 0.0

  # Init gear state
  audioState.currentGear = 1 # Start in 1st gear
  audioState.gearShiftTimer = 0.0

  # Init music state
  audioState.musicPlaying = false
  audioState.musicPosition = 0
  audioState.musicVolume = DEFAULT_MUSIC_VOLUME
  audioState.musicChannels = 0

proc audioShutdown*() =
  saudio.shutdown()

proc updateEngineSound*(carSpeed: float32, carAccel: float32, isDrifting: bool, debugSpeed: var float32, debugRpm: var float32, debugGear: var int32) =
  # Map car speed to target RPM (adjust these ranges to taste)
  const minRpm = 1000.0 # Idle RPM
  const maxRpm = 6000.0 # Redline RPM

  # --- 1. Update Gear Shift Timer ---
  if audioState.gearShiftTimer > 0.0:
    audioState.gearShiftTimer -= sapp.frameDuration()

  # --- 2. Gear Shifting Logic ---
  if audioState.gearShiftTimer <= 0.0:
    # Upshift: When RPM is high and we are accelerating
    # --- Changed > to >= to allow shifting at the exact UPSHIFT_RPM ---
    if audioState.currentRpm >= UPSHIFT_RPM and carAccel > 0.1 and audioState.currentGear < NUM_GEARS:
      audioState.currentGear += 1
      audioState.gearShiftTimer = GEAR_SHIFT_COOLDOWN
      # Drop RPM to simulate the gear change
      audioState.currentRpm = DOWNSHIFT_RPM + 500.0

    # Downshift: When RPM is low or braking
    elif audioState.currentRpm < DOWNSHIFT_RPM and audioState.currentGear > 1:
      if carSpeed > 1.0 or (carAccel < -0.5 and carSpeed > 5.0):
        audioState.currentGear -= 1
        audioState.gearShiftTimer = GEAR_SHIFT_COOLDOWN
        # Jump RPM to simulate the downshift
        audioState.currentRpm = (UPSHIFT_RPM + DOWNSHIFT_RPM) / 2

  # --- 3. Calculate Target RPM based on Gear and Speed ---
  let currentGear = audioState.currentGear
  let minGearSpeed = GEAR_MIN_SPEED[currentGear]
  let maxGearSpeed = GEAR_MAX_SPEED[currentGear]
  let gearSpeedRange = max(0.001, maxGearSpeed - minGearSpeed) # Avoid div by zero

  # Calculate how far "through" the current gear we are based on speed
  # --- Removed clamp() to allow for "over-revving", making shifting more reliable ---
  let speedInGearNormalized = (abs(carSpeed) - minGearSpeed) / gearSpeedRange

  # Map this normalized value to the RPM range for the gear
  var baseTargetRpm = minRpm + (UPSHIFT_RPM - minRpm) * speedInGearNormalized

  # Optional: Log the values to debug
  echo(&"SpeedNorm: {speedInGearNormalized:3.2f} BaseRPM: {baseTargetRpm:5.1f} Vel: {carSpeed:3.1f} - Gear: {currentGear}")
  # Temporary here for debug ui
  debugSpeed = carSpeed
  debugRpm = baseTargetRpm
  debugGear = currentGear.int32

  # --- Final RPM adjustments (Drift, etc.) ---
  if isDrifting: baseTargetRpm = min(maxRpm, baseTargetRpm + DRIFT_RPM_BOOST)
  audioState.targetRpm = clamp(baseTargetRpm, minRpm, maxRpm)

  # Smoothly interpolate current RPM towards target RPM for less jarring changes
  const rpmSmoothingFactor = 4.0 # How fast currentRpm catches up to targetRpm
  audioState.currentRpm = lerp(audioState.currentRpm, audioState.targetRpm, clamp(sapp.frameDuration() * rpmSmoothingFactor, 0.0, 1.0))

  # Convert RPM to a base frequency (fundamental)
  const freqPerRpm = 0.025 # Tune this to get desired pitch range
  audioState.engineFrequency = audioState.currentRpm * freqPerRpm

  # --- Dynamic Sound Shaping ---
  # Normalize RPM to a 0-1 range for blending harmonic levels
  let rpmNormalized = (audioState.currentRpm - minRpm) / (maxRpm - minRpm)
  let blendFactor = clamp(rpmNormalized, 0.0, 1.0)

  # Interpolate harmonic volumes between low and high RPM states
  for i in 0..<MAX_HARMONICS:
    audioState.harmonicsVolumes[i] = lerp(HARMONIC_LEVELS_LOW_RPM[i], HARMONIC_LEVELS_HIGH_RPM[i], blendFactor)

  # Noise increases with RPM (more mechanical noise at higher revs)
  audioState.engineNoiseVolume = 0.05 + 0.15 * blendFactor
  audioState.engineNoiseVolume = clamp(audioState.engineNoiseVolume, 0.0, 0.2) # Clamp noise to prevent it dominating

  # Modulate overall volume based on acceleration and speed
  const baseVolume = 0.2
  const accelVolumeBoost = 0.5 # How much acceleration impacts volume
  const minSpeedVolume = 0.1 # Minimum volume even when stopped

  # Basic volume scales with speed
  let speedVolume = minSpeedVolume + (1.0 - minSpeedVolume) * clamp(abs(carSpeed) / GAME_MAX_SPEED, 0.0, 1.0)

  # Boost volume on acceleration
  let finalVolume = speedVolume + clamp(carAccel * accelVolumeBoost, 0.0, 0.8)
  audioState.engineVolume = clamp(finalVolume * baseVolume, 0.0, 1.0)

  # --- ADD DRIFTING SCREECH VOLUME LOGIC ---
  let targetScreechVolume = if isDrifting and abs(carSpeed) > 0.5: MAX_SCREECH_VOLUME else: 0.0
  let smoothingSpeed = if isDrifting: SCREECH_ATTACK_SPEED else: SCREECH_DECAY_SPEED
  audioState.currentScreechVolume = lerp(audioState.currentScreechVolume, targetScreechVolume, clamp(sapp.frameDuration() * smoothingSpeed, 0.0, 1.0))
  # --- END DRIFTING SCREECH VOLUME LOGIC ---

proc loadMusic*(fs: rtfs.RuntimeFS, filename: string) =
  let musicDataOpt = fs.get(filename)
  if musicDataOpt.isNone:
    echo(&"Could not load music file: {filename}")
    return

  # The QOA decoder needs proc closures that operate on a stream
  var stream = newStringStream(musicDataOpt.get())
  let readByteProc = proc(): int =
    if stream.atEnd: return -1
      #else: return stream.readByte().int
    else: return stream.readUint8().int
  let seekToByteProc = proc(pos: int) =
    stream.setPosition(pos)

  let decoder = newQOADecoder(readByteProc, seekToByteProc)

  if not decoder.readHeader():
    echo(&"Failed to read QOA header for: {filename}")
    stream.close()
    return

  audioState.musicChannels = decoder.getChannels()
  let musicSampleRate = decoder.getSampleRate()
  let totalSamples = decoder.getTotalSamples()

  if musicSampleRate != SAUDIO_SAMPLE_RATE.int:
    echo(&"Music sample rate ({musicSampleRate} Hz) does not match audio device rate ({SAUDIO_SAMPLE_RATE} Hz). This will cause pitch issues. Resampling is not implemented.")

  audioState.musicPcm.setLen(totalSamples * audioState.musicChannels)
  var samplesRead = 0
  while not decoder.isEnd():
    let frameSamples = decoder.readFrame(audioState.musicPcm.toOpenArray(
      samplesRead * audioState.musicChannels,
      audioState.musicPcm.len - 1
    ))
    if frameSamples < 0:
      echo(&"Error decoding QOA frame for: {filename}")
      stream.close()
      return
    samplesRead += frameSamples

  stream.close()

  if samplesRead > 0:
    audioState.musicPlaying = true
    audioState.musicPosition = 0
    echo(&"Successfully loaded music: {filename} ({totalSamples} samples, {audioState.musicChannels} channels)")
  else:
    echo(&"Music file '{filename}' loaded but contains no samples.")

proc audioGenerateSamples*() =
  let expectedFrames = saudio.expect()
  if expectedFrames == 0: return

  let framesToGenerate = min(expectedFrames, SAUDIO_NUM_SAMPLES)

  # --- DYNAMIC LOW-PASS FILTER SETUP for engine ---
  let rpmNormalized = clamp((audioState.currentRpm - 1000.0) / (6000.0 - 1000.0), 0.0, 1.0)
  let filterCutoff = lerp(0.1, 0.8, rpmNormalized)

  for i in 0..<framesToGenerate:
    # --- 1. Generate Engine Sample ---
    var engineSample: float32 = 0.0
    block generateEngine:
      # Sum up harmonic components
      for h in 0..<MAX_HARMONICS:
        let harmonicSample = (2.0 * (audioState.oscillatorPhases[h] - floor(audioState.oscillatorPhases[h])) - 1.0)
        engineSample += harmonicSample * audioState.harmonicsVolumes[h]
        audioState.oscillatorPhases[h] += (audioState.engineFrequency * HARMONIC_FREQS[h]) / SAUDIO_SAMPLE_RATE
        if audioState.oscillatorPhases[h] >= 1.0: audioState.oscillatorPhases[h] -= 1.0
        elif audioState.oscillatorPhases[h] < 0.0: audioState.oscillatorPhases[h] += 1.0

      # Add engine noise
      engineSample += (rand(-1.0..1.0) * audioState.engineNoiseVolume)

      # Add screech noise
      if audioState.currentScreechVolume > 0.001:
        let currentScreechFreq = SCREECH_BASE_FREQ + (rand(-1.0..1.0) * SCREECH_FREQ_JITTER)
        let screechTone = (2.0 * (audioState.screechPhase - floor(audioState.screechPhase)) - 1.0)
        let screechNoise = rand(-1.0..1.0)
        let mixedScreechSample = (screechTone * (1.0 - SCREECH_NOISE_MIX) + screechNoise * SCREECH_NOISE_MIX)
        engineSample += mixedScreechSample * audioState.currentScreechVolume
        audioState.screechPhase += currentScreechFreq / SAUDIO_SAMPLE_RATE
        if audioState.screechPhase >= 1.0: audioState.screechPhase -= 1.0
        elif audioState.screechPhase < 0.0: audioState.screechPhase += 1.0

      # Apply overall engine volume and low-pass filter
      engineSample *= audioState.engineVolume
      engineSample = lerp(audioState.lastOutputSample, engineSample, filterCutoff)
      audioState.lastOutputSample = engineSample

    # --- 2. Generate Music Sample ---
    var musicSample: float32 = 0.0
    block generateMusic:
      if audioState.musicPlaying and audioState.musicPcm.len > 0:
        let pcmLen = audioState.musicPcm.len
        if audioState.musicChannels == 1: # Mono
          musicSample = float32(audioState.musicPcm[audioState.musicPosition]) / 32767.0
          audioState.musicPosition = (audioState.musicPosition + 1)
        elif audioState.musicChannels >= 2: # Stereo or more, mix to mono
          let left = float32(audioState.musicPcm[audioState.musicPosition]) / 32767.0
          let right = float32(audioState.musicPcm[audioState.musicPosition + 1]) / 32767.0
          musicSample = (left + right) * 0.5 # Simple mono mixdown
          audioState.musicPosition += audioState.musicChannels

        # Loop music
        if audioState.musicPosition >= pcmLen:
          audioState.musicPosition = 0

        musicSample *= audioState.musicVolume

    # --- 3. Mix and Clamp ---
    let finalSample = engineSample + musicSample
    audioSamples[i] = clamp(finalSample, -1.0, 1.0)

  discard saudio.push(addr(audioSamples[0]), framesToGenerate)
