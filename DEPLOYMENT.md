# Deployment Guide for Render

This guide will help you deploy the Food Waste Prediction app on Render.

## Prerequisites
- A GitHub account with your code repository
- A Render account (free tier available at https://render.com)

## Deployment Steps

### Option 1: Automatic Deployment (Recommended)
Render will automatically detect the `render.yaml` configuration file.

1. **Sign up/Login to Render**
   - Go to https://render.com
   - Sign up or login with your GitHub account

2. **Create New Web Service**
   - Click "New +" button
   - Select "Web Service"
   - Connect your GitHub repository: `Food-Waste-Prediction-`

3. **Configure Service** (if not auto-detected)
   - **Name**: food-waste-prediction
   - **Environment**: Python 3
   - **Build Command**: `./build.sh`
   - **Start Command**: `gunicorn app:app`

4. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment to complete (5-10 minutes)
   - Your app will be live at: `https://food-waste-prediction-<random>.onrender.com`

### Option 2: Manual Configuration

1. Go to Render Dashboard
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Fill in the details:
   - **Name**: food-waste-prediction
   - **Environment**: Python 3
   - **Region**: Choose closest to you
   - **Branch**: main
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Free

5. Click "Create Web Service"

## Important Notes

- **First Deployment**: May take 5-10 minutes due to installing dependencies
- **Free Tier**: Service may spin down after inactivity (will auto-restart on request)
- **Model Files**: Ensure your `models/` directory is committed to GitHub
- **Environment Variables**: None required for basic setup

## Troubleshooting

### Build Fails
- Check that all files in `models/` and `data/` are committed to Git
- Verify `requirements.txt` has all dependencies

### App Crashes
- Check Render logs in the dashboard
- Ensure `gunicorn` is in `requirements.txt`

### Model Not Found
- Make sure `models/food_waste_model.pkl` exists in your repository
- Check file paths in `config.py`

## Post-Deployment

Once deployed, you can:
- View logs in Render dashboard
- Set up custom domain (paid plans)
- Configure automatic deployments on git push
- Monitor performance metrics

Your app will be accessible at: `https://your-service-name.onrender.com`

---

For more information, visit [Render Documentation](https://render.com/docs)
