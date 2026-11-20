# Call Transcriber - Windows installer.
# Double-click install.cmd (it runs this script). Safe to run again to update.
#   1. installs uv (Python manager) if missing
#   2. installs Python 3.12 + the server's packages into server\.venv
#   3. with an NVIDIA GPU: adds the GPU speech engine (PyTorch + CUDA)
#   4. downloads the speech model
#   5. installs the desktop app (Electron, with Discord built in)
#   6. creates Start-menu and desktop shortcuts

param([switch]$NoPrompt)   # -NoPrompt: unattended (no questions, no windows opened)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Py   = Join-Path $Root 'server\.venv\Scripts\python.exe'
$Desk = Join-Path $Root 'app\desktop'
$Exe  = Join-Path $Desk 'node_modules\electron\dist\electron.exe'
$Icon = Join-Path $Root 'app\windows\icon.ico'

function Step($text) { Write-Host ''; Write-Host "==> $text" -ForegroundColor Cyan }

Step "Installing Call Transcriber in $Root"

# 1. uv --------------------------------------------------------------------
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Step 'Installing uv (Python package manager from astral.sh)'
    powershell -NoProfile -ExecutionPolicy Bypass -Command 'irm https://astral.sh/uv/install.ps1 | iex'
    $env:Path = (Join-Path $env:USERPROFILE '.local\bin') + ';' + $env:Path
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw 'uv did not install. Open a new terminal and run install.cmd again.'
    }
}

# 2. Python + packages -----------------------------------------------------
Step 'Installing Python 3.12 and the transcription server (first time: a few minutes)'
Push-Location (Join-Path $Root 'server')
try {
    $env:UV_PYTHON_PREFERENCE = 'only-managed'   # our own Python, not whatever else is installed (conda etc.)
    uv sync --inexact   # --inexact: keep the GPU engine (torch) installed below, not in the lock file
    if ($LASTEXITCODE -ne 0) { throw 'Installing the server packages failed (see the messages above).' }
} finally { Pop-Location }

# 3. NVIDIA GPU (optional) -------------------------------------------------
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    Step 'NVIDIA GPU found: adding the GPU speech engine (about 3 GB download)'
    $ErrorActionPreference = 'Continue'
    # Pick the PyTorch CUDA build the driver supports (CUDA 12.8 needs driver >= 570).
    $driver = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
    $major = 0
    if ($driver -match '^(\d+)') { $major = [int]$Matches[1] }
    $index = 'https://download.pytorch.org/whl/cu128'
    if ($major -gt 0 -and $major -lt 570) { $index = 'https://download.pytorch.org/whl/cu126' }
    Write-Host "NVIDIA driver $driver - using $index"
    uv pip install --python $Py torch torchaudio --index-url $index
    if ($LASTEXITCODE -eq 0) { uv pip install --python $Py 'transformers>=5' }
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'GPU engine could not be installed; the CPU engine will be used instead (still fast).' -ForegroundColor Yellow
    }
    $ErrorActionPreference = 'Stop'
}

# 4. Speech model ----------------------------------------------------------
Step 'Downloading the speech model'
$env:PYTHONUTF8 = '1'
$load = "import sys; sys.argv=['x']; sys.path.insert(0, r'$Root\server'); import server as s; " +
        "e = s.pick_engine('auto'); print('Speech engine:', e, '-', s.ENGINE_DEFAULTS[e]); " +
        "s.load_engine(e, s.ENGINE_DEFAULTS[e], 'en'); print('Model ready.')"
& $Py -c $load
if ($LASTEXITCODE -ne 0) {
    Write-Host 'The model will be downloaded the first time the app starts instead.' -ForegroundColor Yellow
}

# 5. Desktop app ----------------------------------------------------------
Step 'Installing the app (with Discord built in)'
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host 'Installing Node.js (needed once to set up the app)'
        winget install --id OpenJS.NodeJS.LTS -e --silent --accept-source-agreements --accept-package-agreements
        $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
    }
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { throw 'Node.js is needed: install it from https://nodejs.org and run install.cmd again.' }
}
Push-Location $Desk
try {
    $ErrorActionPreference = 'Continue'
    npm ci --no-audit --no-fund --loglevel=error
    $ErrorActionPreference = 'Stop'
    if (-not (Test-Path $Exe)) { throw 'Installing the app failed (see the messages above).' }
} finally { Pop-Location }

# 6. Shortcuts -------------------------------------------------------------
Step 'Creating Start menu and desktop shortcuts'
$shell = New-Object -ComObject WScript.Shell
$places = @([Environment]::GetFolderPath('Programs'), [Environment]::GetFolderPath('Desktop'))
foreach ($dir in $places) {
    $lnk = $shell.CreateShortcut((Join-Path $dir 'Call Transcriber.lnk'))
    $lnk.TargetPath = $Exe
    $lnk.Arguments = '"' + $Desk + '"'
    $lnk.WorkingDirectory = $Root
    $lnk.IconLocation = $Icon
    $lnk.Description = 'Live transcripts of Discord voice calls'
    $lnk.Save()
}

Write-Host ''
Write-Host 'Done. Start "Call Transcriber" from the Start menu or desktop, sign in to Discord in its Discord window (once), and join a voice channel.' -ForegroundColor Green
if ($NoPrompt) { return }
$answer = Read-Host 'Start Call Transcriber now? (Y/n)'
if ($answer -ne 'n' -and $answer -ne 'N') { Start-Process $Exe -ArgumentList ('"' + $Desk + '"') -WorkingDirectory $Root }
