# Deploying TrustPRO (free tiers)

| Part | Host | Notes |
|---|---|---|
| Database | **Neon** (Postgres, free) | Only rows: users, profiles, reports' metadata. |
| Backend | **Hugging Face Spaces** (Docker SDK, free CPU basic) | Models download on start-up. Files (videos, photos, reports) live on the Space disk, which is **not permanent**: a restart or sleep clears them. |
| UI | **Vercel** (free) | `/api/*` is rewritten to the Space, so cookies stay same-origin. |

## 1. Neon
1. Create a project and copy the connection string.
2. Change the scheme to the driver we use:
   `postgresql+psycopg://USER:PASSWORD@HOST/DB?sslmode=require`
   No manual SQL is needed. On start-up the backend creates the `epsoft` schema and runs every migration.

## 2. Hugging Face Space (backend)
1. huggingface.co → New Space → SDK **Docker** (blank), hardware **CPU basic (free)**. Make it **private**.
2. Space → Settings → **Variables and secrets** → add these as *secrets*:

   | Name | Value |
   |---|---|
   | `DATABASE_URL` | the Neon URL from step 1 |
   | `DB_SCHEMA` | `epsoft` |
   | `ID_NUMBER_PEPPER` | a long random string |
   | `COOKIE_SECURE` | `true` |
   | `PUBLIC_APP_URL` | `https://<your-app>.vercel.app` (the recruiter login link) |
   | `CORS_ORIGINS` | `https://<your-app>.vercel.app` |

3. Push this folder to the Space, using an HF access token with *write* access as the password:
   ```powershell
   cd E:\trustumate\trustpro_backend
   git remote add hf https://huggingface.co/spaces/<hf-user>/<space-name>
   git push hf main
   ```
4. The Space builds from the `Dockerfile` (Python packages only). On first start it downloads YuNet, ArcFace and YOLO11m, exports YOLO11m to OpenVINO and warms up OCR. This takes a few minutes. Watch the **Logs** tab for `Model ready: environment`.
5. Check `https://<hf-user>-<space-name>.hf.space/api/health`. It should show `"database": true`.

The recruiter login (link, login ID, password) for each finished report is printed in the Space **Logs**.

## 3. Vercel (UI)
1. In `trustpro_ui/vercel.json`, replace `REPLACE-WITH-YOUR-SPACE` with `<hf-user>-<space-name>`. Commit and push.
2. vercel.com → Add New Project → import the `TrustPRO_UI` GitHub repo. Vercel detects Vite (build `npm run build`, output `dist`).
3. Open the Vercel URL. Camera access works because Vercel serves HTTPS.

## Limits to know
- **Free CPU basic Space:** 2 vCPU, so reports are slower than on the dev laptop. The Space sleeps after a period without use; the first request after that waits for it to wake up and reload models.
- **Files are not permanent:** after a restart, recordings, photos and old HTML reports are gone, but the database rows remain. Do the assessment and read the report in the same session.
- **Uploads:** they go through Vercel, which limits request bodies to about 4.5 MB. Images are capped at 4 MB; video is uploaded in ~0.5 MB parts.
- **Licences:** the ArcFace weights are non-commercial only. Keep the Space private and use it for demos.
