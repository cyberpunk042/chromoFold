$ErrorActionPreference = 'Stop'
$Source = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$HomeDir = if ($env:CHROMOFOLD_HOME) { $env:CHROMOFOLD_HOME } else { Join-Path $env:LOCALAPPDATA 'ChromoFold' }
$BinDir = Join-Path $HomeDir 'bin'
New-Item -ItemType Directory -Force -Path $HomeDir, $BinDir | Out-Null
foreach ($name in @('tools','hub','product','docs')) {
    $target = Join-Path $HomeDir $name
    if (Test-Path $target) { Remove-Item -Recurse -Force $target }
    Copy-Item -Recurse -Force (Join-Path $Source $name) $target
}
@"
@echo off
python "$HomeDir\tools\chromofold.py" %*
"@ | Set-Content -Encoding ASCII (Join-Path $BinDir 'chromofold.cmd')
@"
@echo off
python "$HomeDir\tools\chromofold_assistant.py" %*
"@ | Set-Content -Encoding ASCII (Join-Path $BinDir 'chromofold-assistant.cmd')
@"
@echo off
python "$HomeDir\hub\server.py" %*
"@ | Set-Content -Encoding ASCII (Join-Path $BinDir 'chromofold-hub.cmd')
Write-Host "Installed ChromoFold in $HomeDir"
Write-Host "Add $BinDir to PATH, then run chromofold-hub."
