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

  # --- New Drift Sound Parameters ---
  DRIFT_RPM_BOOST = 600.0        # How much RPM boosts when drifting
  MAX_SCREECH_VOLUME = 0.6       # INCREASE THIS: Max volume of the tire screech (try 0.8 or 1.0)
  SCREECH_ATTACK_SPEED = 15.0    # INCREASE THIS: How fast screech volume ramps up (make it punchier)
  SCREECH_DECAY_SPEED = 8.0      # INCREASE THIS: How fast screech volume ramps down (so it lingers a bit)

  SCREECH_BASE_FREQ = 1200.0     # Base frequency of the screech sound (e.g., 1200 Hz)
  SCREECH_FREQ_JITTER = 200.0    # How much the screech frequency can randomly vary
  SCREECH_NOISE_MIX = 0.3        # How much white noise to mix with the high-freq tone (0.0 to 1.0)

type
  MusicTrack = object
    name: string
    pcmData: seq[int16]
    channels: int

  AudioState = object
    oscillatorPhases: array[MAX_HARMONICS, float32] # Each harmonic needs its own phase
    screechPhase: float32

    targetRpm: float32
    currentRpm: float32
    engineFrequency: float32
    engineVolume: float32

    harmonicsVolumes: array[MAX_HARMONICS, float32]
    engineNoiseVolume: float32
    currentScreechVolume: float32

    lastOutputSample: float32

    # Music playback state
    playlist: seq[MusicTrack]
    currentTrackIndex: int
    musicPosition: int # Position within the current track
    musicPlaying: bool
    musicVolume: float32

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

  # Init music state
  audioState.playlist = @[]
  audioState.currentTrackIndex = -1
  audioState.musicPlaying = false
  audioState.musicPosition = 0
  audioState.musicVolume = DEFAULT_MUSIC_VOLUME

proc audioShutdown*() =
  saudio.shutdown()

proc updateEngineSound*(carSpeed: float32, carAccel: float32, isDrifting: bool) =
  # Map car speed to target RPM (adjust these ranges to taste)
  const minRpm = 1000.0 # Idle RPM
  const maxRpm = 6000.0 # Max RPM
  const maxSpeed = 20.0 # Max speed of the car for full RPM
  #echo "speed: ", carSpeed, " accel: ", carAccel

  # A simple mapping of speed to a target RPM
  var baseTargetRpm = minRpm + (maxRpm - minRpm) * clamp(abs(carSpeed) / maxSpeed, 0.0, 1.0)

  # --- ADD DRIFT RPM BOOST ---
  if isDrifting:
    baseTargetRpm = min(maxRpm, baseTargetRpm + DRIFT_RPM_BOOST) # Boost RPM, but don't exceed maxRpm
  audioState.targetRpm = baseTargetRpm
  # --- END DRIFT RPM BOOST ---

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
  let speedVolume = minSpeedVolume + (1.0 - minSpeedVolume) * clamp(abs(carSpeed) / maxSpeed, 0.0, 1.0)

  # Boost volume on acceleration
  let finalVolume = speedVolume + clamp(carAccel * accelVolumeBoost, 0.0, 0.8)
  audioState.engineVolume = clamp(finalVolume * baseVolume, 0.0, 1.0)

  # --- ADD DRIFTING SCREECH VOLUME LOGIC ---
  let targetScreechVolume = if isDrifting and abs(carSpeed) > 0.5: MAX_SCREECH_VOLUME else: 0.0
  let smoothingSpeed = if isDrifting: SCREECH_ATTACK_SPEED else: SCREECH_DECAY_SPEED
  audioState.currentScreechVolume = lerp(audioState.currentScreechVolume, targetScreechVolume, clamp(sapp.frameDuration() * smoothingSpeed, 0.0, 1.0))
  # --- END DRIFTING SCREECH VOLUME LOGIC ---

# --- NEW: Stateless decoder helper to avoid the closure-in-a-loop bug ---
proc decodeQoaFromBytes(bytes: seq[byte]): Option[MusicTrack] =
  var pos = 0
  let readByte = proc(): int =
    if pos >= bytes.len: return -1
    result = bytes[pos].int
    inc pos
  let seekByte = proc(newPos: int) =
    pos = clamp(newPos, 0, bytes.len)

  let decoder = newQOADecoder(readByte, seekByte)

  if not decoder.readHeader() or decoder.getSampleRate() != SAUDIO_SAMPLE_RATE.int:
    return none(MusicTrack)

  var track: MusicTrack
  track.channels = decoder.getChannels()
  track.pcmData.setLen(decoder.getTotalSamples * track.channels)
  var samplesRead = 0
  while not decoder.isEnd:
    let frameSamples = decoder.readFrame(track.pcmData.toOpenArray(samplesRead * track.channels, track.pcmData.len - 1))
    if frameSamples < 0: return none(MusicTrack)
    samplesRead += frameSamples

  if samplesRead > 0: return some(track)
  return none(MusicTrack)

proc loadPlaylist*(fs: rtfs.RuntimeFS, directory: string) =
  echo(&"Loading playlist from: {directory}")
  for file in fs.listDir(directory):
    if file.endsWith(".qoa"):
      let fullPath = directory & "/" & file
      let musicDataOpt = fs.get(fullPath)
      if musicDataOpt.isSome:
        var trackOpt = decodeQoaFromBytes(cast[seq[byte]](musicDataOpt.get()))
        if trackOpt.isSome:
          var track = trackOpt.get
          track.name = fullPath # Store the name for logging
          audioState.playlist.add(track)
        else:
          echo(&"Failed to decode QOA file: {fullPath}")
      else:
        echo(&"Could not read file from FS: {fullPath}")

  if audioState.playlist.len == 0:
    echo("No valid music tracks found.")
    return

  randomize(); audioState.playlist.shuffle()
  audioState.currentTrackIndex = 0
  audioState.musicPlaying = true
  echo(&"Playlist loaded with {audioState.playlist.len} tracks.")
  if audioState.playlist[0].pcmData.len > 0:
    echo(&"Now playing: {audioState.playlist[0].name}")

# --- NEW: State management function, called only when needed ---
proc switchToNextTrack() =
  if not audioState.musicPlaying or audioState.playlist.len == 0: return

  var foundNext = false
  # Start search from the track *after* the current one.
  let startIndex = if audioState.currentTrackIndex == -1: 0 else: audioState.currentTrackIndex
  for i in 0 ..< audioState.playlist.len:
    let nextIndex = (startIndex + i + 1) mod audioState.playlist.len
    if audioState.playlist[nextIndex].pcmData.len > 0:
      audioState.currentTrackIndex = nextIndex
      audioState.musicPosition = 0
      #echo(&"Now playing: {audioState.playlist[nextIndex].name}")
      foundNext = true
      break

  if not foundNext:
    audioState.musicPlaying = false
    #echo("No more valid tracks in playlist. Stopping music.")

proc audioGenerateSamples*() =
  let expectedFrames = saudio.expect()
  if expectedFrames == 0: return

  let framesToGenerate = min(expectedFrames, SAUDIO_NUM_SAMPLES)

  # --- 1. STATE MANAGEMENT (runs ONCE per callback) ---
  if audioState.musicPlaying and audioState.playlist.len > 0:
      if audioState.currentTrackIndex == -1:
        switchToNextTrack()
      elif audioState.musicPosition >= audioState.playlist[audioState.currentTrackIndex].pcmData.len:
        switchToNextTrack()

  # --- 2. HOIST LOOKUPS (runs ONCE per callback) ---
  # Get a direct reference to the PCM data *before* the hot loop. This is the critical fix.
  var currentPcm: seq[int16]
  var currentNumChannels = 0
  if audioState.musicPlaying and audioState.currentTrackIndex != -1:
    let track = audioState.playlist[audioState.currentTrackIndex]
    currentPcm = track.pcmData
    currentNumChannels = track.channels
  # --- DYNAMIC LOW-PASS FILTER SETUP for engine ---
  let rpmNormalized = clamp((audioState.currentRpm - 1000.0) / (6000.0 - 1000.0), 0.0, 1.0)
  let filterCutoff = lerp(0.1, 0.8, rpmNormalized)

  # --- 3. GENERATE SAMPLES (The main loop) ---
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
      # Use the 'cached' direct reference inside the loop for maximum performance.
      if currentPcm.len > 0 and audioState.musicPosition < currentPcm.len:
        if currentNumChannels == 1:
          musicSample = float32(currentPcm[audioState.musicPosition]) / 32767.0
          inc audioState.musicPosition
        else: # Stereo mixdown
          if audioState.musicPosition + 1 < currentPcm.len:
            let left = float32(currentPcm[audioState.musicPosition]) / 32767.0
            let right = float32(currentPcm[audioState.musicPosition+1]) / 32767.0
            musicSample = (left + right) * 0.5
          inc(audioState.musicPosition, currentNumChannels)
        musicSample *= audioState.musicVolume

    # --- 3. Mix and Clamp ---
    let finalSample = engineSample + musicSample
    audioSamples[i] = clamp(finalSample, -1.0, 1.0)

  discard saudio.push(addr(audioSamples[0]), framesToGenerate)
