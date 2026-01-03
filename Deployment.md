# 🚀 F1 AI Race Engineer - Deployment Guide

## Local Setup (Docker)

### 1. Install Docker
- **Windows/Mac**: https://www.docker.com/products/docker-desktop
- **Linux**: `sudo apt-get install docker.io docker-compose`

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Build & Run
```bash
make build
make up
```

### 4. Access
- Frontend: http://localhost:3000
- Backend: http://localhost:8000/docs

---

## Free Deployment Options

### Option 1: Railway (Recommended - FREE $5 credit)
1. Sign up: https://railway.app
2. Connect GitHub repo
3. Deploy in 1 click
4. Free tier includes PostgreSQL + Redis

### Option 2: Render (FREE tier)
1. Sign up: https://render.com
2. Create Web Service (backend) - FREE
3. Create Static Site (frontend) - FREE
4. Add PostgreSQL - FREE
5. Add Redis - FREE (via Upstash)

### Option 3: Fly.io (FREE allowance)
1. Sign up: https://fly.io
2. Install CLI: `curl -L https://fly.io/install.sh | sh`
3. Deploy: `fly launch`

---

## Troubleshooting

### Port already in use
```bash
# Stop conflicting services
sudo lsof -i :3000
sudo lsof -i :8000
```

### Docker issues
```bash
# Restart Docker
sudo systemctl restart docker  # Linux
# Or restart Docker Desktop (Windows/Mac)
```