# Start all three services in separate terminals (run from repo root)

Write-Host "Patient Fidelity Demo" -ForegroundColor Cyan
Write-Host ""
Write-Host "Open 3 terminals and run:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Terminal 1 - Avatar worker:" -ForegroundColor Green
Write-Host "    cd services/avatar-worker"
Write-Host "    .venv\Scripts\activate   # after: python -m venv .venv && pip install -r requirements.txt"
Write-Host "    python main.py"
Write-Host ""
Write-Host "  Terminal 2 - Agent:" -ForegroundColor Green
Write-Host "    cd services/agent"
Write-Host "    .venv\Scripts\activate"
Write-Host "    python main.py dev"
Write-Host ""
Write-Host "  Terminal 3 - Web:" -ForegroundColor Green
Write-Host "    cd apps/web"
Write-Host "    npm run dev"
Write-Host ""
Write-Host "Then open http://localhost:3000" -ForegroundColor Cyan
