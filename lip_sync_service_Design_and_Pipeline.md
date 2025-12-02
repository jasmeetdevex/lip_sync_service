# lip_sync_service — Design and Pipeline (brief)

## Snapshot: last acceptable demo
- Commit: 7cb64a0e147351d09d8bb03631c9054a57a3b2ad
- Tag: `lipsync_good_demo` (created and pushed to origin)
- Author date: Thu Nov 13 15:34:19 2025 +0530
- Notes: This commit contains the last *acceptable* demo configuration and command updates for Wav2Lip at the time of the demo (invocation uses the Wav2Lip inference script via the service tasks).

## Current stable snapshot (post-fix)
- Commit: bd1d81c35378b32b0dc0b3e2757de33aae50dbaf
- Author date: Thu Nov 13 17:08:12 2025 +0530
- Reason it's stable: Fixes in `tasks.py` add robust error handling for downloads, VRAM monitoring, retry and timeout handling for inference/ffmpeg, S3 region-correct URL construction, and safer cleanup logic. These changes improved reliability and are considered the most stable lip-sync state now.

Example working `task_id` used in testing: `cdde5238-bdb8-4ae3-aa39-d90aa375177a`

---

## How the earlier demo pipeline was invoked (quick reference)
The earlier demo used the Wav2Lip inference script directly or through the service task. Minimum form:

- Direct (Wav2Lip repo):

  ```bash
  python inference.py --checkpoint_path <checkpoint.pth> --face <input_video.mp4> --audio <input_audio.wav> --outfile <output.mp4>
  ```

- Through the service (Celery task): the service would invoke the same script using the `run_wav2lip_task` Celery task which calls the Wav2Lip `inference.py` using subprocess with absolute paths to the checkpoint, face and audio files and an `--outfile` destination. Basic parameters commonly set by the service include `--fps 25`, `--resize_factor <1|2|4>`, `--pads <t> <b> <l> <r>` and `--nosmooth`.

Operation duration expected: ~1–2 hours (cap 2 hours) for full pipeline runs depending on model and instance resources.

---

## High-level changes since the demo
- Services merged: lip-sync service merged with the upscaling component (see later merge commits) — pipeline now can optionally run an upscaling/enhancement step after lip-sync.
- Tasks architecture:
  - Task model extended to support multiple outputs and models (Wav2Lip non‑GAN + Wav2Lip‑GAN).
  - Celery tasks updated to run both model types, optionally enhance GAN outputs, and store multiple S3 keys / URLs.
  - Improved error handling and retry strategies during download, inference, and upload stages.
- Resource handling and observability:
  - VRAM monitoring and peak/aggregate statistics collected during inference runs.
  - Timeout handling for both inference and ffmpeg to avoid hanging jobs.
  - Logs and stdout/stderr captured per model for debugging.
- Deployment/CI: added checkpoint validation, S3 region-aware upload URLs, and explicit S3_BUCKET_NAME checks to fail faster when environment variables are missing.

---

## Where to look in the repo
- Service entry points: `app.py`, `tasks.py`, `worker.py`
- Model container & inference: `Wav2Lip/inference.py`, `Wav2Lip/checkpoints/` (local references)
- Controllers / API: `Controllers/lip_sync_controller.py`, `routes/lip_sync_routes.py`
- Models: `models/lipSyncTasksModal.py` (task schema changes)

---

## Recommended next steps
1. Run an end-to-end test using `task_id: cdde5238-bdb8-4ae3-aa39-d90aa375177a` to verify end-to-end success on a representative instance.
2. If this fixed state is intended as the canonical stable release, consider tagging the stable commit (bd1d81c) with something like `lipsync_stable_2025-11-13`.
3. Add minimal automated system/integration tests that exercise a full task (download → inference → upload) on CI with mocked external dependencies.

---

If you want, I can also: create a `lipsync_stable` tag at the stable commit, add a small end-to-end test harness, or update any documentation with run commands/examples for the current service behavior.
