# Complete setup guide — Patient Fidelity Demo

Follow these steps **in order**.

---

## Part 1 — Create the GitHub repository (one time)

### On your PC (PowerShell)

```powershell
cd C:\Users\raymi\Desktop\PersonalCoding\Work\work-projects\rai-demo-fidelity

git branch -M main
git commit -m "Initial commit: patient fidelity demo"
```

### On GitHub (browser)

1. Go to https://github.com/new
2. Repository name: **`rai-demo-fidelity`**
3. Set to **Private** (recommended — even without secrets in git, it's a demo project)
4. **Do not** add README, .gitignore, or license (we already have them)
5. Click **Create repository**

### Push from your PC

Replace `YOUR_GITHUB_USERNAME` with your GitHub username:

```powershell
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/rai-demo-fidelity.git
git push -u origin main
```

You may be prompted to sign in to GitHub.

---

## Part 2 — Add API keys on your PC (one time)

Secrets are **not** in GitHub. Copy examples and fill in your keys:

```powershell
cd C:\Users\raymi\Desktop\PersonalCoding\Work\work-projects\rai-demo-fidelity

copy apps\web\.env.example apps\web\.env.local
copy services\agent\.env.example services\agent\.env
copy services\avatar-worker\.env.example services\avatar-worker\.env
```

Edit each file with your LiveKit, OpenAI, and Cartesia keys (see root `.env.example` for variable names).

---

## Part 3 — Install dependencies on your PC (one time)

### Web app

```powershell
cd apps\web
npm install
```

### Voice agent

```powershell
cd C:\Users\raymi\Desktop\PersonalCoding\Work\work-projects\rai-demo-fidelity\services\agent
py -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If `py` doesn't work, use:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv .venv
```

---

## Part 4 — Clone on RunPod (one time per pod)

Open your RunPod pod → **Web Terminal** (this is **Linux bash**, not Windows).

```bash
cd /workspace
git clone https://github.com/YOUR_GITHUB_USERNAME/rai-demo-fidelity.git
cd rai-demo-fidelity/services/avatar-worker
pip install -r requirements.txt
```

Create `.env` on the pod (paste your keys):

```bash
cat > .env << 'EOF'
LIVEKIT_URL=wss://rai-demo-fidelity-dly9lkyg.livekit.cloud
LIVEKIT_API_KEY=APIzoN8XK88zjPh
LIVEKIT_API_SECRET=your-secret-here
LIVEKIT_ROOM=patient-demo
PATIENT_LOOP_PATH=assets/alan-loop.mp4
TARGET_FPS=25
AVATAR_MODE=mock
EOF
```

---

## Part 5 — Run the demo (every time)

Use **3 terminals**. Start in this order.

### Terminal A — RunPod web terminal (avatar video)

```bash
cd /workspace/rai-demo-fidelity/services/avatar-worker
export $(grep -v '^#' .env | xargs)
python main.py
```

Wait for: `Avatar joined room patient-demo`  
**Leave running.**

### Terminal B — Your PC PowerShell (voice agent)

```powershell
cd C:\Users\raymi\Desktop\PersonalCoding\Work\work-projects\rai-demo-fidelity\services\agent
.\.venv\Scripts\activate
python main.py dev
```

Wait for agent worker to connect.  
**Leave running.**

### Terminal C — Your PC PowerShell (web app)

```powershell
cd C:\Users\raymi\Desktop\PersonalCoding\Work\work-projects\rai-demo-fidelity\apps\web
npm run dev
```

Open **http://localhost:3000** → **Start session** → type a message to Alan.

---

## Cheat sheet: Windows vs RunPod

| | Your PC (PowerShell) | RunPod (bash) |
|--|----------------------|---------------|
| Activate venv | `.\.venv\Scripts\activate` | not needed |
| Run Python | `python main.py` | `python main.py` |
| Path separator | `\` | `/` |
| Avatar worker | ❌ don't run here | ✅ run here |
| Agent | ✅ run here | ❌ |
| Web app | ✅ run here | ❌ |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `py` not found on PC | Open a **new** terminal, or use full path to `Python311\python.exe` |
| `bash: .venv\Scripts\activate` on RunPod | You're on Linux — skip venv, just `pip install` and `python main.py` |
| No patient video | RunPod avatar worker must be running first |
| No voice from Alan | Agent must be running on your PC |
| `git clone` asks for password | Use GitHub personal access token as password, or SSH clone URL |

---

## Cost reminder

~**$0.12 per 10-minute session** at $0.69/hr GPU. **Stop your RunPod pod** when not demoing.
