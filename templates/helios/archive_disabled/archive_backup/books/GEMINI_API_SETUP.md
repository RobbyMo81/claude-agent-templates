# **GEMINI API KEY INTEGRATION GUIDE**

## **Quick Setup**

### **Step 1: Get Your API Key**
1. Visit: https://aistudio.google.com/app/apikey
2. Sign in with Google account
3. Click **"Create API Key"**
4. Copy the generated key (starts with `AIzaSy...`)

### **Step 2: Add API Key to Environment**
1. Open `.env.local` in your project root
2. Replace `PLACEHOLDER_API_KEY` with your real API key:
   ```
   VITE_GEMINI_API_KEY=AIzaSyD_your_actual_key_here
   ```

### **Step 3: Rebuild with API Key**
**Option A - PowerShell (Windows):**
```powershell
.\build-with-api-key.ps1
```

**Option B - Manual Docker Build:**
```powershell
docker build --build-arg VITE_GEMINI_API_KEY="AIzaSyD_your_key" -t helios-image .
```

### **Step 4: Run Updated Container**
```powershell
docker stop helios-container; docker rm helios-container
docker run -d --name helios-container -p 8080:8080 helios-image
```

---

## **Security Features**

### **Build-Time Integration**
- API key baked into frontend build (secure)
- No runtime environment variable exposure
- Works in production deployments

### **Local Development**
- `.env.local` kept out of git
- `.env.example` template provided
- Automatic placeholder detection

### **Production Deployment**
- `.gcloudignore` excludes sensitive files
- Build arguments for secure key passing
- No API keys in deployed source code

---

## **Cloud Run Deployment with API Key**

### **Build and Deploy:**
```bash
# Build with API key
gcloud builds submit \
  --substitutions _VITE_GEMINI_API_KEY="YOUR_API_KEY" \
  --tag gcr.io/YOUR-PROJECT/helios

# Deploy to Cloud Run
gcloud run deploy helios \
  --image gcr.io/YOUR-PROJECT/helios \
  --platform managed \
  --allow-unauthenticated
```

### **Using Cloud Build Substitutions:**
Create `cloudbuild.yaml`:
```yaml
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: 
  - 'build'
  - '--build-arg'
  - 'VITE_GEMINI_API_KEY=${_VITE_GEMINI_API_KEY}'
  - '-t'
  - 'gcr.io/$PROJECT_ID/helios'
  - '.'
substitutions:
  _VITE_GEMINI_API_KEY: 'PLACEHOLDER_API_KEY'  # Override via command line
```

---

## **Testing the Integration**

### **Verify API Key Working:**
1. Open http://localhost:8080
2. Upload a CSV file in "Level 0: Baseline" 
3. Click **"Launch Baseline"**
4. Check console - should see no API key errors
5. AI analysis should work properly

### **Expected Results:**
- No "API key not valid" errors
- Gemini AI analysis in test results
- All features fully functional

---

## **Troubleshooting**

### **"API key not valid" Error:**
1. Check `.env.local` has correct key format
2. Verify key starts with `AIzaSy`
3. Rebuild container with `.\build-with-api-key.ps1`

### **Key Not Found in Build:**
1. Ensure using `--build-arg VITE_GEMINI_API_KEY="key"`
2. Check Dockerfile has `ARG VITE_GEMINI_API_KEY` line
3. Verify environment variable names match

### **Production Issues:**
1. Use Cloud Build substitutions for API key
2. Never commit `.env.local` to git
3. Test locally before deploying

---

**Status: 🟢 READY FOR API KEY INTEGRATION**

Once you paste your Gemini API key, run `.\build-with-api-key.ps1` and you'll have a fully functional Helios system with AI-powered analysis!
