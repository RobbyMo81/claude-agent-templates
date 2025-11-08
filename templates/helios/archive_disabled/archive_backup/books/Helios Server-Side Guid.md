Here is a Helios Server-Side Troubleshooting & Reference Guide, tailored to your codebase and workflows:

---

# Helios Server-Side Troubleshooting & Reference Guide

## 1. Server Startup Issues

**Symptoms:**
- Server does not start, or crashes on import.
- Logs show `ImportError` for ML dependencies (e.g., torch, numpy).

**Resolutions:**
- Ensure the Python virtual environment is activated.
- Install all required packages:  
  `pip install -r requirements.txt`  
  (If requirements.txt is missing, install at least: `flask`, `flask_cors`, `sqlalchemy`, `alembic`, `torch`, `numpy`.)
- If running in Docker, rebuild the image to ensure dependencies are present.

---

## 2. API Always Returns Mock Data

**Symptoms:**
- All `/api/train`, `/api/train/status/<job_id>`, and related endpoints return mock responses, regardless of input.
- Logs show forced mock mode.

**Resolutions:**
- In `server.py`, search for `if True:` or similar forced mock switches. Replace with a real mode check (e.g., `if not DEPENDENCIES_AVAILABLE or ...`).
- Ensure all ML dependencies are installed and importable.
- Confirm that the backend is not running in a development/mock configuration.

---

## 3. Database & Migration Problems

**Symptoms:**
- Errors about missing tables, migrations, or database connection.
- Alembic migration scripts not found or not applied.

**Resolutions:**
- Check `alembic.ini` for correct database URL (default is SQLite).
- Run migrations:  
  `alembic upgrade head`
- If Alembic is not initialized, run:  
  `alembic init alembic`
- Ensure models are correctly imported in Alembic’s `env.py`.

---

## 4. Static Files or Frontend Not Served

**Symptoms:**
- Visiting `/` or frontend routes returns 404 or API-only JSON.
- Logs indicate missing static files.

**Resolutions:**
- Build the frontend:  
  `npm run build` (from the frontend directory)
- Ensure the build output is in the expected directory (`../dist` for local, `/app/static` for Docker).
- Check `static_folder_path` logic in `server.py`.

---

## 5. 500 Internal Server Errors

**Symptoms:**
- API endpoints return 500 errors.
- Logs show stack traces or unhandled exceptions.

**Resolutions:**
- Review logs for the specific error and traceback.
- Validate all input data (ensure required fields are present).
- Check that all components (trainer, memory_store, etc.) are initialized and not `None`.
- Add or improve try/except blocks for error handling.

---

## 6. Job Status or History Not Found

**Symptoms:**
- `/api/train/status/<job_id>` returns 404 or always mock data.
- `/api/train/history` is empty or fails.

**Resolutions:**
- Ensure jobs are being persisted (not just mocked).
- Check that the database is up-to-date and the job table exists.
- If using mock mode, disable it for real job tracking.

---

## 7. Analytics & Advanced Endpoints Fail

**Symptoms:**
- `/api/analytics/*` endpoints return incomplete responses or errors.
- Some endpoints have missing return statements or empty branches.

**Resolutions:**
- Complete the implementation of these endpoints.
- Ensure all required dependencies and components are loaded.
- Add error handling and return informative error messages.

---

## 8. General Debugging Steps

1. **Check Logs:** Always review the server logs for error messages and stack traces.
2. **Validate Environment:** Confirm the correct Python environment and all dependencies.
3. **Database Health:** Run Alembic migrations and check DB connectivity.
4. **Frontend Build:** Ensure the frontend is built and static files are in place.
5. **Mock vs. Real Mode:** Know which mode you are running in; disable forced mocks for production.
6. **Component Initialization:** All major components must be initialized and not `None`.
7. **API Testing:** Use Postman, curl, or similar tools to test endpoints.
8. **Refactor Redundancy:** Remove or refactor repeated mock logic and forced mock switches.
9. **Security:** For production, add authentication, input validation, and rate limiting.

---

## 9. Common Error Messages & Quick Fixes

| Error Message / Symptom                | Likely Cause                        | Quick Fix                                      |
|----------------------------------------|-------------------------------------|------------------------------------------------|
| `ImportError: ...`                     | Missing dependency                  | Install package, check venv                    |
| `500 Internal Server Error`            | Unhandled exception                 | Check logs, add try/except, validate input     |
| `404 Not Found` (API)                  | Wrong endpoint or missing resource  | Check route, ensure resource exists            |
| `Frontend not found`                   | Static files missing                | Build frontend, check static path              |
| `Training job not found`               | Job ID invalid or not persisted     | Check DB, disable mock mode                    |
| `Model management not available`       | Dependencies missing or not loaded  | Install dependencies, check initialization     |
| `Cross-model analytics not available`  | Component not initialized           | Check imports and initialization               |

---

## 10. When to Escalate

- If you encounter persistent import or DB errors after following the above, check for:
  - File permission issues (especially in Docker or on Windows).
  - Corrupted virtual environment (try recreating it).
  - Incompatible package versions.

---

## 11. Maintenance & Best Practices

- Keep requirements.txt and Alembic migrations up to date.
- Regularly test both mock and production modes.
- Document any changes to the API or data model.
- Use version control for all code and migration scripts.
- Monitor logs for recurring errors or warnings.

---

**If you follow this guide, you should be able to resolve most server-side issues in Helios. For persistent or complex problems, document the error and steps taken, then escalate to the development lead.**