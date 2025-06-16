# rtfs.nim - A simple runtime filesystem for game assets
import std/os
import std/strutils
import std/strformat
import std/options; export options

type
  RuntimeFS* = object
    rootPath: string # The absolute path to our asset root directory

proc newRuntimeFS*(relativeRoot: string): RuntimeFS =
  ## Creates a new runtime filesystem object.
  ## It uses conditional compilation to handle Emscripten's virtual FS correctly.
  when defined(emscripten):
    # For Emscripten, --preload-file assets creates a root folder called 'assets'.
    # So our asset root path inside the virtual FS is /assets.
    result.rootPath = "/" / relativeRoot
  else:
    # For desktop builds, the path is relative to the executable.
    result.rootPath = getAppDir() / relativeRoot

  echo(&"[rtfs] Initialized with root: {result.rootPath}")

  when not defined(emscripten):
    if not dirExists(result.rootPath):
      echo(&"[rtfs] Warning: Root directory does not exist, creating: {result.rootPath}")
      createDir(result.rootPath)

func looksAbsolute(path: string): bool =
  when doslikeFileSystem:
    path.len >= 3 and path[1..2] == ":\\"
  else:
    path.startsWith("/")

proc getPath*(fs: RuntimeFS, filename: string): Option[string] =
  ## Safely gets the full, absolute path for a filename within the FS.
  ## Returns `none` if the path is invalid or outside the root.
  let targetPath = fs.rootPath / filename

  when defined(emscripten):
    return some(targetPath)
  else:
    if looksAbsolute(filename) or not targetPath.isRelativeTo(fs.rootPath):
      echo(&"[rtfs] Access denied to path outside root: {filename}")
      return none(string)
    return some(targetPath)

proc get*(fs: RuntimeFS, filename: string): Option[string] =
  ## Gets the contents of a file relative to the filesystem's root.
  let targetPathOpt = fs.getPath(filename)
  if targetPathOpt.isNone: return none(string)

  let targetPath = targetPathOpt.get()
  if fileExists(targetPath):
    try:
      let content = readFile(targetPath)
      # --- SUCCESS LOG ---
      echo(&"[rtfs] Successfully loaded: {targetPath} ({content.len} bytes)")
      return some(content)
    except CatchableError as e:
      # --- ERROR LOG ---
      echo(&"[rtfs] Error reading file '{targetPath}': {e.msg}")
      return none(string)
  else:
    # --- NOT FOUND LOG ---
    echo(&"[rtfs] File not found: {targetPath}")
    return none(string)

proc write*(fs: RuntimeFS, filename: string, content: string | seq[byte]) =
  ## Writes content to a file. This is disabled for Emscripten builds.
  when not defined(emscripten):
    let targetPathOpt = fs.getPath(filename)
    if targetPathOpt.isNone:
      echo(&"[rtfs] Cannot write file, invalid path: {filename}")
      return

    let targetPath = targetPathOpt.get()
    try:
      let parentDir = targetPath.parentDir()
      if not dirExists(parentDir):
        createDir(parentDir)
      writeFile(targetPath, content)
      echo(&"[rtfs] Wrote file: {targetPath}")
    except CatchableError as e:
      echo(&"[rtfs] Error writing file '{targetPath}': {e.msg}")
  else:
    echo(&"[rtfs] Warning: Write operation skipped in Emscripten build for file: {filename}")


proc fileExists*(fs: RuntimeFS, filename: string): bool =
  ## Checks if a file exists within the filesystem.
  let targetPathOpt = fs.getPath(filename)
  if targetPathOpt.isNone: return false
  let exists = os.fileExists(targetPathOpt.get())
  # --- FILEEXISTS LOG ---
  echo(&"[rtfs] Checking exists '{targetPathOpt.get()}': {exists}")
  return exists

iterator walk*(fs: RuntimeFS): string =
  ## Lists all file names relative to the root, recursively.
  for path in walkDirRec(fs.rootPath, relative=true):
    yield path

iterator listDir*(fs: RuntimeFS, subdir = ""): string =
  ## Lists files and directories in a subdirectory relative to the root.
  let targetPathOpt = fs.getPath(subdir)
  if targetPathOpt.isNone:
    echo "Not found targetPath listDir"

  let targetDir = targetPathOpt.get()
  echo(&"[rtfs] Listing directory: {targetDir}")
  if dirExists(targetDir):
    for item in walkDir(targetDir):
      yield item.path.relativePath(targetDir)
