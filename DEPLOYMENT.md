# Research Insight Assistant - Deployment Guide

## Quick Start (Local with Docker)

```bash
# 1. Clone and configure
git clone <your-repo>
cd research-insight-assistant
cp .env.example .env
# Edit .env with your OPENROUTER_APIKEY

# 2. Start with Docker Compose
docker-compose up -d

# 3. Access
# Frontend: http://localhost:8501
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

---

## Platform-Specific Deployment

### 1. Railway (Recommended - Easiest)

Railway supports both Docker and native Python deployments.

**Option A: Docker (recommended)**
```bash
# 1. Install Railway CLI
npm i -g @railway/cli

# 2. Login and init
railway login
railway init

# 3. Deploy
railway up

# 4. Set environment variables in Railway dashboard:
# OPENROUTER_APIKEY=your-key
# OPENROUTER_MODEL=anthropic/claude-sonnet-4
```

**Option B: Native Python**
- Connect your GitHub repo to Railway
- Set build command: `pip install -r requirements.txt`
- Set start command: `python -m uvicorn api:app --host 0.0.0.0 --port $PORT`
- Add a second service for Streamlit with start command: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`

---

### 2. Render

**Backend (Web Service)**
```yaml
# render.yaml
services:
  - type: web
    name: research-assistant-api
    env: docker
    dockerfilePath: ./Dockerfile
    envVars:
      - key: OPENROUTER_APIKEY
        sync: false
      - key: OPENROUTER_MODEL
        value: anthropic/claude-sonnet-4
    autoDeploy: true

  - type: web
    name: research-assistant-frontend
    env: docker
    dockerfilePath: ./Dockerfile
    buildCommand: ""
    startCommand: streamlit run streamlit_app.py --server.port=8080 --server.address=0.0.0.0 --server.headless=true
    envVars:
      - key: API_URL
        value: https://research-assistant-api.onrender.com
    autoDeploy: true
```

Deploy: Push to GitHub → Connect repo in Render → Services created automatically.

---

### 3. Fly.io

```bash
# 1. Install flyctl
curl -L https://fly.io/install.sh | sh

# 2. Launch (creates fly.toml)
fly launch --dockerfile Dockerfile

# 3. Set secrets
fly secrets set OPENROUTER_APIKEY=your-key
fly secrets set OPENROUTER_MODEL=anthropic/claude-sonnet-4

# 4. Deploy
fly deploy
```

For Streamlit, create a second app or use a process manager in a single container.

---

### 4. Google Cloud Run

```bash
# 1. Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/research-assistant

# 2. Deploy API
gcloud run deploy research-assistant-api \
  --image gcr.io/PROJECT_ID/research-assistant \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENROUTER_APIKEY=your-key,OPENROUTER_MODEL=anthropic/claude-sonnet-4

# 3. Deploy Frontend (separate service)
gcloud run deploy research-assistant-frontend \
  --image gcr.io/PROJECT_ID/research-assistant \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars API_URL=https://research-assistant-api-xxx.run.app \
  --command "streamlit run streamlit_app.py --server.port=8080 --server.address=0.0.0.0 --server.headless=true"
```

---

### 5. AWS ECS/Fargate

Use the `docker-compose.yml` with AWS Copilot or ECS ComposeX:
```bash
# With AWS Copilot
copilot init --name research-assistant --dockerfile Dockerfile
copilot env init --name production --default-config
copilot deploy --env production
```

---

### 6. DigitalOcean App Platform

Create `.do/app.yaml`:
```yaml
name: research-assistant
services:
- name: api
  source_dir: /
  github:
    repo: your-org/research-insight-assistant
    branch: main
  run_command: python -m uvicorn api:app --host 0.0.0.0 --port 8080
  environment_slug: python
  instance_count: 1
  instance_size_slug: basic-xxs
  envs:
  - key: OPENROUTER_APIKEY
    value: your-key
    type: SECRET
  - key: OPENROUTER_MODEL
    value: anthropic/claude-sonnet-4
  http_port: 8080
  health_check:
    http_path: /health
- name: frontend
  source_dir: /
  github:
    repo: your-org/research-insight-assistant
    branch: main
  run_command: streamlit run streamlit_app.py --server.port=8080 --server.address=0.0.0.0 --server.headless=true
  environment_slug: python
  instance_count: 1
  instance_size_slug: basic-xxs
  envs:
  - key: API_URL
    value: ${api.PUBLIC_URL}
  http_port: 8080
```

---

## Production Considerations

### 1. Database Persistence
- **ChromaDB**: Uses local file storage. For production, mount a persistent volume.
- **SQLite**: Conversation DB is local. Consider PostgreSQL for multi-instance deployments.

### 2. Scaling
- **API**: Stateless, can scale horizontally behind a load balancer
- **Streamlit**: Single-user sessions, sticky sessions needed or run single instance
- **ChromaDB**: Single-writer, consider managed vector DB (Pinecone, Weaviate) for scale

### 3. Secrets Management
- Never commit `.env` to git
- Use platform secret managers (Railway, Render, Fly.io, AWS Secrets Manager, etc.)

### 4. Monitoring
- Add logging: `structlog` or Python `logging`
- Add metrics: Prometheus client + Grafana
- Health checks already configured at `/health`

### 5. Custom Domain + SSL
All platforms above provide automatic HTTPS with custom domains.

---

## Cost Estimates (Monthly)

| Platform | API + Frontend | Notes |
|----------|---------------|-------|
| Railway | $5-20 | Simple, includes DB |
| Render | $7-15 | Free tier available |
| Fly.io | $5-25 | Pay for what you use |
| Cloud Run | $0-10 | Very cheap for low traffic |
| DigitalOcean | $12-24 | Predictable pricing |
| AWS ECS | $15-50 | More complex |

---

## CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to Railway
        uses: railway/cli@v3
        with:
          command: up
        env:
          RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ChromaDB permission errors | Ensure volume mounts have correct permissions |
| Model download timeout | Increase startup timeout, pre-download models |
| Out of memory | Use smaller models, increase container memory |
| Streamlit can't connect to API | Check `API_URL` env var, network connectivity |
| PDF upload fails | Check file size limits, volume mounts |

---

## Next Steps

1. **Add authentication** - Integrate Auth0, Clerk, or custom JWT
2. **Add rate limiting** - Prevent abuse with `slowapi` or similar
3. **Managed vector DB** - Migrate to Pinecone/Weaviate for production scale
4. **PostgreSQL** - Replace SQLite for conversation persistence
5. **CDN** - Serve static assets via CloudFlare/AWS CloudFront