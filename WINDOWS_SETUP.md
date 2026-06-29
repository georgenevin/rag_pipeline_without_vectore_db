# Windows Setup Guide for Infopark RAG System

## Step-by-Step Setup on Windows

### Step 1: Verify Python Installation

Open PowerShell and verify Python is installed:
```powershell
python --version
# Should show Python 3.8 or higher
```

If you get an error, [install Python](https://www.python.org/downloads/) and add it to PATH.

### Step 2: Create Virtual Environment (Recommended)

```powershell
# Navigate to project directory
cd "C:\Users\Nevin George\projects\Scrapping"

# Create virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1
```

If you get an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```powershell
# Option A: Using pip
pip install -r requirements.txt

# Option B: Using uv (if installed)
uv add -r requirements.txt

# Verify installation
pip list
```

### Step 4: Set Mistral API Key

#### Method 1: PowerShell (Temporary - Current Session Only)
```powershell
$env:MISTRAL_API_KEY = "your-mistral-api-key-here"
```

#### Method 2: PowerShell (Permanent - All Sessions)
```powershell
# This sets it permanently for your user
[Environment]::SetEnvironmentVariable("MISTRAL_API_KEY", "your-api-key", "User")

# Verify it was set
$env:MISTRAL_API_KEY
# Should show your API key

# You may need to restart PowerShell for it to take effect
```

#### Method 3: Command Prompt
```cmd
setx MISTRAL_API_KEY "your-api-key"
```

#### Method 4: GUI Method
1. Press `Win + X` and select "System"
2. Click "Advanced system settings"
3. Click "Environment Variables"
4. Under "User variables", click "New"
5. Variable name: `MISTRAL_API_KEY`
6. Variable value: `your-api-key`
7. Click OK and restart applications

**Important:** Restart PowerShell/CMD after setting environment variables!

### Step 5: Get Mistral API Key

1. Go to [Mistral AI Console](https://console.mistral.ai/)
2. Sign up or log in
3. Navigate to "API Keys"
4. Create new API key
5. Copy the key (you'll only see it once!)

### Step 6: Verify Setup

```powershell
# Check API key is set
Write-Host $env:MISTRAL_API_KEY

# Test Python can find packages
python -c "import langchain; print('LangChain OK')"

# Test RAG system can load
python -c "from rag import EmbeddingRAG; print('RAG OK')"
```

### Step 7: Run RAG System

```powershell
# Interactive mode
python main.py

# Or quick query
python query.py "What is Infopark?"
```

## Troubleshooting on Windows

### Issue 1: "ModuleNotFoundError"

**Problem:** `ModuleNotFoundError: No module named 'langchain'`

**Solution:**
```powershell
# Make sure virtual environment is activated
.\.venv\Scripts\Activate.ps1

# Reinstall packages
pip install --upgrade -r requirements.txt
```

### Issue 2: "Python not recognized"

**Problem:** `'python' is not recognized as an internal or external command`

**Solution:**
```powershell
# Use full path
C:\Users\YourUsername\AppData\Local\Programs\Python\Python312\python.exe query.py "test"

# Or add Python to PATH in Environment Variables
```

### Issue 3: "MISTRAL_API_KEY not set"

**Problem:** `ValueError: MISTRAL_API_KEY environment variable not set`

**Solution:**
```powershell
# Verify it's set
Write-Host $env:MISTRAL_API_KEY

# If empty, set it
$env:MISTRAL_API_KEY = "your-key"

# For permanent, use GUI method (Step 4, Method 4)
```

### Issue 4: "Permission denied" for Script Activation

**Problem:** `File cannot be loaded because running scripts is disabled`

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

### Issue 5: "SSL: CERTIFICATE_VERIFY_FAILED"

**Problem:** Network/SSL certificate error

**Solution:**
```powershell
# Update certificates
python -m certifi

# Or reinstall with:
pip install --upgrade certifi

# Then retry your command
```

### Issue 6: Slow Performance

**Problem:** Queries take too long

**Solution:**
```powershell
# Use fewer documents
python query.py -k 2 "your question"

# Or check network:
Test-NetConnection -ComputerName console.mistral.ai -Port 443
```

## VSCode Integration on Windows

### Install Python Extension
1. Open VSCode
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "Python"
4. Install Microsoft's Python extension

### Select Python Interpreter
1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose `.\.venv\Scripts\python.exe`

### Create VSCode Tasks

Create `.vscode/tasks.json`:
```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Run RAG Interactive",
      "type": "shell",
      "command": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
      "args": ["main.py"],
      "group": {
        "kind": "build",
        "isDefault": true
      }
    },
    {
      "label": "Run Query",
      "type": "shell",
      "command": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
      "args": ["query.py", "--examples"]
    }
  ]
}
```

Then press `Ctrl+Shift+B` to run tasks.

## Windows Terminal Setup

### Add Project to Windows Terminal

1. Open Windows Terminal settings (Ctrl+,)
2. Add a new profile:
```json
{
  "name": "Infopark RAG",
  "commandline": "powershell.exe",
  "startingDirectory": "C:\\Users\\Nevin George\\projects\\Scrapping"
}
```

### Quick Launch Script

Create `launch_rag.bat`:
```batch
@echo off
cd "C:\Users\Nevin George\projects\Scrapping"
.venv\Scripts\activate.bat
python main.py
pause
```

Then double-click to run!

## Common Windows Paths

```
Project:      C:\Users\Nevin George\projects\Scrapping
Python Exec:  C:\Users\Nevin George\projects\Scrapping\.venv\Scripts\python.exe
Embeddings:   C:\Users\Nevin George\projects\Scrapping\embeddings.jsonl
```

## Batch File for Setup

Create `setup.bat`:
```batch
@echo off
echo Setting up Infopark RAG System...

cd /d "%~dp0"

echo Installing dependencies...
pip install -r requirements.txt

echo Setup complete!
echo.
echo Next steps:
echo 1. Set MISTRAL_API_KEY environment variable
echo 2. Run: python main.py
pause
```

Run with: `setup.bat`

## Batch File for Query

Create `run_query.bat`:
```batch
@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"
.venv\Scripts\activate.bat

set /p query="Enter your query: "
python query.py "%query%"

pause
```

## Quick Verification Checklist

- [ ] Python installed: `python --version`
- [ ] Virtual environment activated: `.venv\Scripts\Activate.ps1`
- [ ] Packages installed: `pip list` shows langchain, mistralai, etc.
- [ ] API key set: `Write-Host $env:MISTRAL_API_KEY`
- [ ] Embeddings file exists: `Test-Path embeddings.jsonl`
- [ ] Can import RAG: `python -c "from rag import EmbeddingRAG"`
- [ ] Can run: `python main.py`

## Next Steps

```powershell
# Start interactive mode
python main.py

# Or test with a quick query
python query.py "What is Infopark?"

# Or see all examples
python query.py --examples
```

## Need More Help?

- **README_RAG.md** - Full documentation
- **RAG_USAGE.md** - Detailed usage guide  
- **QUICK_REFERENCE.md** - Quick commands
- **rag_examples.py** - Working code examples

---

**Happy querying! 🚀**
