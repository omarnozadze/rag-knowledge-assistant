$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\.venv")) {
    python -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

if (-not (Test-Path ".\.env")) {
    Copy-Item ".\.env.example" ".\.env"
    Write-Host ""
    Write-Host "Created .env. Add OPENAI_API_KEY and run this script again."
    exit 1
}

& ".\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8020 --reload
