# rtfs.nim - A simple runtime filesystem for game assets
import std/os
import std/strutils
import std/options; export options

type
  RuntimeFS* = object
    rootPath: string # The absolute path to our asset root directory

proc newRuntimeFS*(relativeRoot: string): RuntimeFS =
  ## Creates a new runtime filesystem object.
  ## The `relativeRoot` path is resolved relative to the application's executable.
  result.rootPath = getAppDir() / relativeRoot
  echo "[rtfs] Initialized with root: " & result.rootPath
  # Ensure the root directory exists
  if not dirExists(result.rootPath):
    echo "[rtfs] Warning: Root directory does not exist, creating: " & result.rootPath
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
  if looksAbsolute(filename) or not targetPath.isRelativeTo(fs.rootPath):
    echo "[rtfs] Access denied to path outside root: " & filename
    return none(string)
  return some(targetPath)

proc get*(fs: RuntimeFS, filename: string): Option[string] =
  ## Gets the contents of a file relative to the filesystem's root.
  let targetPathOpt = fs.getPath(filename)
  if targetPathOpt.isNone: return none(string)

  let targetPath = targetPathOpt.get()
  if fileExists(targetPath):
    try:
      return some(readFile(targetPath))
    except CatchableError as e:
      echo "[rtfs] Error reading file '", targetPath, "': ", e.msg
      return none(string)
  else:
    return none(string)

# ADD THIS NEW `write` PROCEDURE
proc write*(fs: RuntimeFS, filename: string, content: string | seq[byte]) =
  ## Writes content to a file relative to the filesystem's root.
  let targetPathOpt = fs.getPath(filename)
  if targetPathOpt.isNone:
    echo "[rtfs] Cannot write file, invalid path: " & filename
    return

  let targetPath = targetPathOpt.get()
  try:
    # Ensure parent directory exists before writing
    let parentDir = targetPath.parentDir()
    if not dirExists(parentDir):
      createDir(parentDir)
    writeFile(targetPath, content)
    echo "[rtfs] Wrote file: " & targetPath
  except CatchableError as e:
    echo "[rtfs] Error writing file '", targetPath, "': ", e.msg

proc fileExists*(fs: RuntimeFS, filename: string): bool =
  ## Checks if a file exists within the filesystem.
  let targetPathOpt = fs.getPath(filename)
  if targetPathOpt.isNone: return false
  return os.fileExists(targetPathOpt.get())

iterator walk*(fs: RuntimeFS): string =
  ## Lists all file names relative to the root, recursively.
  for path in walkDirRec(fs.rootPath, relative=true):
    yield path

iterator listDir*(fs: RuntimeFS, subdir = ""): string =
  ## Lists files and directories in a subdirectory relative to the root.
  let targetPathOpt = fs.getPath(subdir)
  if targetPathOpt.isNone:
    echo "Not found"

  let targetDir = targetPathOpt.get()
  if dirExists(targetDir):
    for item in walkDir(targetDir):
      yield item.path.relativePath(targetDir)
