# 🚀 Complete GitHub Repository Setup Guide
## TTC-Security-Attacks - Step-by-Step PowerShell Commands

---

## 📋 BEFORE YOU BEGIN

### What You Need:
1. **Git** installed on your computer
2. **GitHub account** (you have: https://github.com/hmshujaatzaheer)
3. **PowerShell** (comes with Windows) or Terminal (Mac/Linux)

### Check if Git is installed:
```powershell
git --version
```
If you see a version number like `git version 2.43.0`, you're good!
If not, download Git from: https://git-scm.com/downloads

---

## 🎯 STEP 1: Create Repository on GitHub Website

### 1.1 Open GitHub in your browser
```
Go to: https://github.com/hmshujaatzaheer
```

### 1.2 Click the green "New" button (top right, next to "Repositories")

### 1.3 Fill in the details:
```
Repository name: ttc-security-attacks
Description: Mechanistic Attacks on Test-Time Compute: Exploiting Step-Level Verification, Single-Model Voting, and Tree Search in Reasoning LLMs
```

### 1.4 Settings to select:
- ✅ Public (so everyone can see your research)
- ❌ Do NOT check "Add a README file" (we have our own)
- ❌ Do NOT add .gitignore (we have our own)
- ❌ Do NOT choose a license (we have our own)

### 1.5 Click "Create repository"

---

## 🎯 STEP 2: Download the Repository Files

First, you need to download the repository files I created. You have two options:

### Option A: If you have the files from Claude
Copy all the files to a folder on your computer, for example:
```
C:\Users\YourName\Projects\ttc-security-attacks
```

### Option B: Download from this session
Ask me to create a ZIP file of all the repository files.

---

## 🎯 STEP 3: Open PowerShell

### 3.1 Press Windows Key + X
### 3.2 Click "Windows PowerShell" or "Terminal"

You should see something like:
```
PS C:\Users\YourName>
```

---

## 🎯 STEP 4: Navigate to Your Project Folder

### 4.1 Go to where you want to create the project
```powershell
# Change to your preferred directory
cd C:\Users\YourName\Projects

# If the folder doesn't exist, create it first:
mkdir C:\Users\YourName\Projects
cd C:\Users\YourName\Projects
```

### 4.2 Create the project folder (if not already created)
```powershell
mkdir ttc-security-attacks
cd ttc-security-attacks
```

### 4.3 Verify you're in the right place
```powershell
pwd
```
Should show: `C:\Users\YourName\Projects\ttc-security-attacks`

---

## 🎯 STEP 5: Initialize Git Repository

### 5.1 Initialize a new Git repository
```powershell
git init
```
You should see: `Initialized empty Git repository in ...`

### 5.2 Configure your Git identity (if not done before)
```powershell
git config --global user.name "H M Shujaat Zaheer"
git config --global user.email "shujabis@gmail.com"
```

### 5.3 Verify configuration
```powershell
git config --global --list
```

---

## 🎯 STEP 6: Create All Project Files

Now create all the files. Copy each file's content into the appropriate location.

### 6.1 Create folder structure
```powershell
# Create all directories
mkdir src\attacks
mkdir src\defenses
mkdir src\evaluation
mkdir src\utils
mkdir configs
mkdir data\datasets
mkdir data\results
mkdir docs
mkdir notebooks
mkdir tests
mkdir scripts
```

### 6.2 Verify structure
```powershell
tree /F
```

### 6.3 Create/copy all files
You need to create each file. Here's how for one example:

```powershell
# Example: Create README.md using notepad
notepad README.md
```
Then paste the content and save.

**Or use VS Code (recommended):**
```powershell
# Open the folder in VS Code
code .
```
Then create all files in VS Code.

---

## 🎯 STEP 7: Add Files to Git

### 7.1 Check current status
```powershell
git status
```
You'll see all files in red (untracked).

### 7.2 Add ALL files to staging
```powershell
git add .
```

### 7.3 Verify files are staged
```powershell
git status
```
Now files should be in green (staged).

---

## 🎯 STEP 8: Create First Commit

### 8.1 Commit all files with a message
```powershell
git commit -m "Initial commit: TTC Security Attacks framework

- Implemented NLBA (Natural Language Blindness Attack) for PRMs
- Implemented SMVA (Single-Model Voting Attack) for self-consistency  
- Implemented MVNA (MCTS Value Network Attack) for tree search
- Added TTC-Sec benchmark framework
- Added configuration files and documentation
- Added comprehensive README with usage examples"
```

### 8.2 Verify commit
```powershell
git log --oneline
```
You should see your commit.

---

## 🎯 STEP 9: Connect to GitHub

### 9.1 Add GitHub as remote origin
```powershell
git remote add origin https://github.com/hmshujaatzaheer/ttc-security-attacks.git
```

### 9.2 Verify remote
```powershell
git remote -v
```
Should show:
```
origin  https://github.com/hmshujaatzaheer/ttc-security-attacks.git (fetch)
origin  https://github.com/hmshujaatzaheer/ttc-security-attacks.git (push)
```

---

## 🎯 STEP 10: Push to GitHub

### 10.1 Rename branch to main (if needed)
```powershell
git branch -M main
```

### 10.2 Push to GitHub
```powershell
git push -u origin main
```

### 10.3 Enter credentials when prompted
- **Username**: hmshujaatzaheer
- **Password**: Your GitHub Personal Access Token (NOT your password!)

---

## 🔑 STEP 10.5: Creating a Personal Access Token (if needed)

If prompted for password and regular password doesn't work:

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Give it a name: "TTC Security Attacks Push"
4. Select scopes: ✅ repo (all)
5. Click "Generate token"
6. COPY THE TOKEN IMMEDIATELY (you won't see it again!)
7. Use this token as your password

---

## 🎯 STEP 11: Verify on GitHub

### 11.1 Open your repository in browser
```
https://github.com/hmshujaatzaheer/ttc-security-attacks
```

### 11.2 You should see:
- README.md displayed beautifully
- All folders and files
- Green checkmark next to files

---

## 📝 COMPLETE COMMAND SUMMARY

Here's everything in one block you can copy-paste:

```powershell
# ===== ONE-TIME SETUP (if Git not configured) =====
git config --global user.name "H M Shujaat Zaheer"
git config --global user.email "shujabis@gmail.com"

# ===== NAVIGATE TO PROJECT FOLDER =====
cd C:\Users\YourName\Projects\ttc-security-attacks

# ===== INITIALIZE GIT =====
git init

# ===== ADD ALL FILES =====
git add .

# ===== COMMIT =====
git commit -m "Initial commit: TTC Security Attacks framework"

# ===== CONNECT TO GITHUB =====
git remote add origin https://github.com/hmshujaatzaheer/ttc-security-attacks.git

# ===== PUSH =====
git branch -M main
git push -u origin main
```

---

## 🔄 FUTURE UPDATES

When you make changes later:

```powershell
# 1. Add changes
git add .

# 2. Commit with message
git commit -m "Your descriptive message here"

# 3. Push to GitHub
git push
```

---

## ❓ TROUBLESHOOTING

### "fatal: not a git repository"
→ Run `git init` first

### "remote origin already exists"
→ Run `git remote remove origin` then add again

### "failed to push some refs"
→ Run `git pull origin main --rebase` then push again

### "Permission denied"
→ Check your Personal Access Token

---

## 📊 Repository Information

| Property | Value |
|----------|-------|
| **Repository Name** | ttc-security-attacks |
| **Full URL** | https://github.com/hmshujaatzaheer/ttc-security-attacks |
| **Clone URL** | https://github.com/hmshujaatzaheer/ttc-security-attacks.git |
| **Owner** | hmshujaatzaheer |
| **License** | MIT |
| **Language** | Python |

---

## 📝 Repository Description for GitHub

Copy this for the "About" section:

```
Mechanistic Attacks on Test-Time Compute: Exploiting Step-Level Verification, Single-Model Voting, and Tree Search in Reasoning LLMs. Includes NLBA, SMVA, MVNA attacks and TTC-Sec benchmark.
```

### Topics to Add (on GitHub):
```
machine-learning, adversarial-attacks, llm-security, process-reward-models, 
self-consistency, mcts, reasoning-models, test-time-compute, ai-safety
```
This repositry was orchasterated with the aid of modern tooling to enhance clarity, structure and reproductibility. 

---

**🎉 Congratulations! Your research repository is now live on GitHub!**
