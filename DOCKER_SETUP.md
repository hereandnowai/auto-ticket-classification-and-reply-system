# 🐳 Docker Setup Guide — Shanyan AI Ticket System

Run the entire system (**backend + frontend + ML model**) with a single command using Docker. No Python, Node.js, or any other setup required — just Docker.

---

## Prerequisites

1. **Install Docker Desktop for Windows**
   - Download from [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
   - During installation, ensure **WSL 2** backend is enabled (recommended)
   - After installation, start Docker Desktop and wait for it to say "Docker is running"

2. **Install Git** (if not already installed)
   - Download from [https://git-scm.com/downloads](https://git-scm.com/downloads)

---

## Quick Start (3 Steps)

### Step 1: Clone the Repository

Open **Command Prompt** or **PowerShell** and run:

```bash
git clone https://github.com/hereandnowai/auto-ticket-classification-and-reply-system.git
cd auto-ticket-classification-and-reply-system
```

### Step 2: Create Your `.env` File

Copy the example env file and add your Gemini API key:

```bash
copy .env.example .env
```

Then open `.env` in Notepad and replace `your_gemini_api_key_here` with your actual API key:

```
GEMINI_API_KEY=your_actual_api_key_here
```

> 💡 Get a free API key at [https://aistudio.google.com/api-keys](https://aistudio.google.com/api-keys)

### Step 3: Start Everything

```bash
docker compose up --build
```

> ⏳ **First run takes 15-30 minutes** (downloads ML model + trains it). Subsequent starts are instant.

---

## Access the Application

Once you see `🎉 Backend is running!` in the logs:

| Service         | URL                                                      |
| --------------- | -------------------------------------------------------- |
| **Frontend**    | [http://localhost:5173](http://localhost:5173)           |
| **Backend API** | [http://localhost:8000](http://localhost:8000)           |
| **API Docs**    | [http://localhost:8000/docs](http://localhost:8000/docs) |

### Demo Accounts

| Role         | Username  | Password    |
| ------------ | --------- | ----------- |
| Admin        | `admin`   | `admin123`  |
| Client       | `client1` | `client123` |
| Tech Support | `tech1`   | `tech123`   |
| Accounting   | `acc1`    | `acc123`    |
| Sales        | `sales1`  | `sales123`  |

---

## Common Commands

```bash
# Start the system (first time or after changes)
docker compose up --build

# Start in background (detached mode)
docker compose up -d

# View logs
docker compose logs -f

# View only backend logs
docker compose logs -f backend

# Stop the system
docker compose down

# Stop and remove all data (models + database)
docker compose down -v

# Rebuild a specific service
docker compose build backend
docker compose build frontend
```

---

## Troubleshooting

### "Docker is not running"

Make sure Docker Desktop is started and shows "Docker is running" in the bottom bar.

### First startup seems stuck

Model training can take 15-30 minutes. Watch the backend logs:

```bash
docker compose logs -f backend
```

You'll see progress messages like "Training DistilBERT model..." and training epochs.

### Port already in use

If port 5173 or 8000 is in use, either stop the other process or change the ports in `docker-compose.yml`:

```yaml
ports:
  - "3000:80" # Change frontend from 5173 to 3000
  - "9000:8000" # Change backend from 8000 to 9000
```

### Need to retrain the model

```bash
docker compose down -v   # Remove volumes (including trained model)
docker compose up --build
```

---

Created by **Shankar Narayanan**, Student, Dr MGR University
