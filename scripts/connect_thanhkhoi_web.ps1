param([string]$ConfigPath = (Join-Path (Split-Path -Parent $PSScriptRoot) 'configs\thanhkhoi-web.json'))
$config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
if ($config.user -ne 'thanhkhoi' -or $config.host -notmatch '^[a-zA-Z0-9.-]+$' -or
    $config.remote_root -notmatch '^/mnt/thanhkhoi/[a-zA-Z0-9_-]+$') {
    throw 'Unexpected SSH destination or remote working directory'
}
$destination = '{0}@{1}' -f $config.user, $config.host
$sshArguments = @('-p', [string]$config.port, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes')
& ssh @sshArguments $destination "sh '$($config.remote_root)/scripts/start_thanhkhoi_api.sh' --background"
if ($LASTEXITCODE -ne 0) { throw 'Could not start the thanhkhoi API; inspect the SSH error.' }
$forward = '127.0.0.1:{0}:127.0.0.1:{1}' -f $config.local_api_port, $config.remote_api_port
Write-Host "Forwarding localhost:$($config.local_api_port) to the thanhkhoi GPU API. Keep this terminal open."
& ssh @sshArguments '-o' 'ExitOnForwardFailure=yes' '-N' '-L' $forward $destination
exit $LASTEXITCODE
