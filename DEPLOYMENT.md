# Render Deployment with Docker

## Option 1: Using Docker (RECOMMENDED - Most Reliable)

This method guarantees Python 3.11.9 and all dependencies work correctly.

### Steps:

1. **In your Render dashboard:**
   - Click "New +" → "Web Service"
   - Connect your GitHub repo: `Krishna1129/Food-Waste-Prediction-`

2. **Configure service:**
   - **Name**: `food-waste-prediction`
   - **Environment**: `Docker` (NOT Python!)
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Dockerfile Path**: `Dockerfile` (should auto-detect)
   - **Docker Build Context Directory**: `.` (root)
   
3. **No build or start commands needed**
   - Docker handles everything automatically

4. **Click "Create Web Service"**

### Why Docker?
- ✅ Guarantees Python 3.11.9
- ✅ Pre-built wheels install without issues
- ✅ Faster builds
- ✅ More reliable deployment
- ✅ No version conflicts

---

## Option 2: Python (Native) - If Docker Doesn't Work

If you must use native Python environment:

### In Render Dashboard:
- **Environment**: `Python 3`
- **Build Command**: 
  ```bash
  pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
  ```
- **Start Command**: 
  ```bash
  gunicorn app:app
  ```
- **Environment Variables** (add this):
  - Key: `PYTHON_VERSION`
  - Value: `3.11.9`

---

## Troubleshooting

### Current Issues:
1. **Python version not being detected correctly**
2. **Pandas failing to build from source**

### Solution:
**Use Docker** (Option 1) - This completely bypasses Python version detection issues.

---

## After Deployment:

Your app will be at: `https://food-waste-prediction.onrender.com` (or similar)

Test the prediction endpoint to ensure it works!
