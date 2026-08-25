param([string]$Output = "submission.zip")
$required = @("script.py", "features.py", "requirements.txt", "model/catboost.cbm", "model/metadata.json")
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing required file: $path" }
}
Compress-Archive -LiteralPath script.py,features.py,requirements.txt,model -DestinationPath $Output -Force
Write-Host "Created $Output"
