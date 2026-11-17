param(
    [ValidateSet('cpu', 'cuda')][string]$Device = 'cuda',
    [ValidateRange(1, 65535)][int]$Port = 8000
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv-notebook\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Missing .venv-notebook. Follow docs/official_v2_web_validation.md to provision it.'
}
$env:CATBREED_MODEL = 'real'
$env:CATBREED_MODEL_PACKAGE = Join-Path $projectRoot 'model_package'
$env:CATBREED_DEVICE = $Device
$env:CATBREED_PRECISION = 'float32'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& $python -m uvicorn app.api.main:app --app-dir $projectRoot --host 127.0.0.1 --port $Port
exit $LASTEXITCODE
