# Portable Docker installation and sharing

## What is packaged

The same Dockerfile builds two Linux images for `linux/amd64` (Intel/AMD 64-bit):

- `demo`: the integrated browser UI and deterministic simulated backends. This
  is the default for `docker build .` and `docker compose up`.
- `realtime`: the integrated browser UI with CPU PyTorch, faster-whisper,
  Transformers/Qwen/SUPERB, TensorFlow/DeepFace, Piper, and eSpeak NG. It uses
  the existing real CPU configuration and inference settings.

There is no separate Windows image. Docker Desktop runs the Linux container
inside WSL 2 on Windows 11. Linux users run the same image with Docker Engine.
Python 3.11 and the application dependencies live inside the image; nobody needs
your Python installation, virtual environment, Windows username, or file paths.

Docker is the packaging mechanism, not a guarantee of model quality or a public
hosting security boundary. This remains a research prototype without
authentication. Compose publishes only `127.0.0.1`, not your LAN or the internet.

## Prerequisites

- **Windows 11:** [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/),
  WSL 2 backend, hardware virtualization enabled, and Linux containers selected.
- **Linux:** [Docker Engine](https://docs.docker.com/engine/install/) and the
  [Compose plugin](https://docs.docker.com/compose/install/linux/). Follow the
  Docker documentation for permissions; membership in the `docker` group grants
  root-equivalent access to the machine.
- Git or a downloaded source ZIP, a current browser, and internet access for the
  first build. Real mode also needs internet access for initial weight downloads.
- Real-mode planning budget: at least 8 GiB of Docker-available RAM, preferably
  12–16 GiB, and 15–20 GiB of free disk. Actual needs depend on package/model
  versions and concurrent applications. CPU inference can take tens of seconds.

Check from a terminal:

```text
docker version
docker compose version
```

Use Compose v2 or newer (`docker compose`, with a space). These files do not
require PowerShell, Bash scripts, CUDA, privileged containers, or device mounts.
ARM/aarch64 machines can try amd64 emulation, but performance and compatibility
are not validated. This is not a GPU image.

## Get the source

After the owner publishes the repository, replace the example URL:

```text
git clone https://github.com/YOUR_USERNAME/project-phantom.git
cd project-phantom
```

Alternatively, select **Code > Download ZIP** on GitHub, extract it, and open a
terminal in the extracted folder containing both `Dockerfile` and
`docker-compose.yml`. Upload the application folder's contents at the GitHub
repository root, not as a nested `project-phantom/` subdirectory; GitHub must see
`.github/workflows/` at the root to run the workflows.

## Packaging verification status (2026-10-03)

The handoff checks used a fresh Windows Python 3.11 environment: **259 core
tests passed**, one optional-ML module was skipped, and 17 optional/slow tests
were deselected. Lint, formatting, strict typing, and the configured Bandit scan
passed. Both Compose profiles parsed with the official Compose CLI. A live local
demo HTTP server passed the browser-assets/text-turn/session-deletion smoke
check. Linux amd64/Python 3.11 dependency resolution and package wheel building
also passed.

**No Docker Engine was available for this handoff.** Actual Linux image builds,
container execution, and full real-model startup were not run here. The GitHub
Docker jobs are added, not already executed. Run those builds and the real-mode
acceptance checks before calling this a cross-platform-validated release.

## Run the lightweight demo

```text
docker compose up --build -d
docker compose ps
docker compose exec phantom python scripts/container_smoke.py --mode demo
```

Open **http://127.0.0.1:8000/**. Demo output is intentionally synthetic and
speech is text-only. No real model is downloaded or invoked. The smoke check
checks the UI assets, six demo status entries, text-only consent, a generated
text turn, deletion, and refusal to reuse the deleted session. It uses no camera,
microphone, participant text, or personal media.

```text
docker compose logs -f phantom
docker compose down
```

`Ctrl+C` ends log viewing, not the detached container. `down` stops the app and
clears all in-process sessions.

## Run the full CPU application

Stop the demo, then merge the real-mode override with the base file:

```text
docker compose down
docker compose -f docker-compose.yml -f compose.realtime.yaml up --build -d
docker compose -f docker-compose.yml -f compose.realtime.yaml logs -f phantom
```

Use the same browser URL. On first start, the entrypoint downloads and verifies
the pinned Arabic Piper voice. The app then warms all six real backends and
downloads missing weights through their existing adapters. This can take
several minutes or longer; the health check has a 30-minute startup allowance.
A download/checksum failure does not enable demo mode. Look at the logs if the
container restarts or stays unavailable.

After startup:

```text
docker compose -f docker-compose.yml -f compose.realtime.yaml ps
docker compose -f docker-compose.yml -f compose.realtime.yaml exec phantom python scripts/container_smoke.py --mode real
```

The smoke check requires all six backends to report `ready` and verifies a real
text response plus a WAV response. It does not test speech-to-text accuracy,
facial-expression validity, Arabic dialect performance, camera permissions, or
clinical safety. Test the browser's camera, microphone, Arabic transcript
correction, **Read aloud**, and **Stop & delete** controls manually.

Arabic/mixed replies use the same pinned Piper voice. English-only replies use
eSpeak NG inside the Linux image, including when the host is Windows. English
voice quality differs from native Windows SAPI. Sound plays in the browser,
not on a host audio device opened by the container.

Stop real mode with the same file selection:

```text
docker compose -f docker-compose.yml -f compose.realtime.yaml down
```

To switch back to demo, stop real mode first and run the base demo command.
Compose uses one service named `phantom`; do not start both profiles concurrently
on the same port.

## Model caches and privacy

- A named volume, normally `project-phantom_phantom-models`, mounts at
  `/var/lib/phantom` in real mode. It holds Hugging Face caches, DeepFace weights,
  and verified Piper files under `tts/`. It contains software artifacts, not
  session recordings or transcripts.
- Containers run as UID/GID `10001:10001`, with a read-only root filesystem,
  dropped Linux capabilities, `no-new-privileges`, resource limits, and a
  memory-backed `/tmp` for transient upload/synthesis/configuration files.
- A generated temporary config uses absolute container paths for Piper. Your
  source YAML, inference parameters, safety configuration, and Windows model
  directory are not rewritten.
- Models remain cached across restarts and `down`. Sessions, raw uploaded media,
  and conversation history are not saved into that volume. Stopping/deleting
  sessions cannot remove independent browser, OS, screenshot, or proxy copies.
- Do not mount a personal dataset, your user profile, a camera, `/dev/snd`, the
  Docker socket, or a host virtual environment into this container.

**Cache deletion is optional and destructive.** The following command removes
the real-mode model volume. All downloaded weights must be downloaded again:

```text
docker compose -f docker-compose.yml -f compose.realtime.yaml down --volumes
```

After weights are cached, inference is local. Some upstream download checks may
still need network access; this setup does not promise fully offline cold starts.

## Another port or resource limit

No `.env` is required. To override Compose settings, create `.env` beside the
Compose files (or copy `.env.example`) and set, for example:

```dotenv
PHANTOM_PORT=8001
PHANTOM_REAL_MEMORY=8g
PHANTOM_REAL_CPUS=4.0
```

Then recreate the selected profile with `up -d`. Open
`http://127.0.0.1:8001/`. These are Compose settings, not Python launcher flags.
Compose deliberately sets the config file for each image; a host `.env` cannot
accidentally turn the demo image into real mode. It does not forward API keys or
the entire host environment. Do not change the loopback binding to `0.0.0.0`.

## Build and share an image archive

GitHub stores the source and Docker recipe. Image archives are optional release
artifacts, not files to commit into Git. A recipient can build the image locally
instead of receiving your existing container. Never distribute a saved running
container: it may contain user/session artifacts.

Build and export the **demo** image:

```text
docker build --platform linux/amd64 --target demo -t project-phantom:demo .
docker save -o project-phantom-demo.tar project-phantom:demo
```

Send the archive together with `docker-compose.yml`, or use a GitHub Release
attachment if it fits GitHub's current size limits. On the recipient's machine:

```text
docker load -i project-phantom-demo.tar
docker compose up -d --no-build
```

This avoids a local build but still requires Docker. The `.tar` archive is ignored
by Git. Use `docker save`, not `docker export`; export loses image configuration.

For a locally built real image:

```text
docker build --platform linux/amd64 --target realtime -t project-phantom:realtime .
```

**Before sending or publishing the real binary image**, review
[third-party speech licenses](third_party_tts.md) and every bundled dependency's
redistribution obligations. The image contains GPL-licensed Piper and eSpeak NG;
PHANTOM's MIT license alone does not cover them. Once obligations are satisfied,
the same `docker save`/`docker load` method applies with tag
`project-phantom:realtime` and both Compose files. Weight caches are not part of
the image archive. Recipients still download their own weights on first start
under the upstream model terms. The safest initial public release is source plus
the demo image, with instructions for local real-image builds.

## Optional prebuilt image on GitHub Container Registry

`.github/workflows/docker.yml` builds both images on Linux and checks:

- merged Compose configuration;
- the running hardened demo container and integrated browser/text/session flow;
- real-image dependency imports, CPU-only PyTorch, and genuine headless English
  WAV synthesis without network access or model downloads.

It does **not** automatically publish packages, download pretrained weights, or
claim end-to-end real model validation. The separate core CI tests Python
3.11–3.13 on Windows and Linux.

To publish a prebuilt **demo** image after the checks pass:

1. Push the source to your own GitHub repository, with workflows at its root.
2. Open **Actions > Docker > Run workflow** on your reviewed branch.
3. Enable `publish_demo` and run the workflow. It uses GitHub's scoped
   `GITHUB_TOKEN`; no personal access token belongs in your source or `.env`.
4. Set the resulting package's visibility to **public** if you want anonymous
   users to pull it. Public repository visibility does not automatically imply
   public package visibility.

The image names are `ghcr.io/YOUR_USERNAME/YOUR_REPOSITORY:demo` and
`:demo-COMMIT_SHA` (repository path lowercased). Recipients replace the example:

```text
docker run -d --name phantom-demo --init --read-only --tmpfs /tmp:rw,size=256m,mode=1777,noexec,nosuid,nodev --cap-drop ALL --security-opt no-new-privileges:true --pids-limit 256 --memory 1g --cpus 2 --publish 127.0.0.1:8000:8000 ghcr.io/YOUR_USERNAME/YOUR_REPOSITORY:demo
docker exec phantom-demo python scripts/container_smoke.py --mode demo
docker stop phantom-demo
docker rm phantom-demo
```

Use the commit-specific tag for a fixed tested build. No image or GitHub
repository has been published by generating these files. Real binary-image
publishing is deliberately left for a separate dependency-license review.

## GitHub source release checklist

Publish only the application folder, not its parent folder containing unrelated
proposals, presentations, archives, or participant materials.

Check the repository boundary **before staging files**:

```text
git rev-parse --show-toplevel
```

The result should be the folder containing `Dockerfile`, `pyproject.toml`, and
`.github/`. If it reports a parent folder, stop: `git add .` at that parent would
include unrelated files. To create a separate application repository without
removing or resetting the parent repository, run the following **inside the
application folder**, only if it does not already have its own `.git` directory:

```text
git init -b main
```

This creates a nested, independent repository; it does not erase parent history.
Alternatively, copy only the source into a new clean folder outside that parent
repository and initialize it there. Recheck `git rev-parse --show-toplevel`.

Then review and commit the source:

```text
git status --short --untracked-files=all
git add .
git diff --cached --stat
git diff --cached
git commit -m "chore: prepare portable Docker release"
```

- Confirm `.env`, keys, `.venv*`, caches, model weights, datasets, media, logs,
  image archives, and participant information are absent from the staged diff.
  `.gitignore` cannot untrack a file that was previously committed.
- Check source/docs manually for credentials and personal information. Ignore
  rules are not a complete secret scanner. Rotate any key ever exposed.
- `.dockerignore` uses an allowlist, so local environments, private media,
  datasets, weights, and unrelated docs do not enter the build context.
- Include hidden source files: `.github/`, `.gitignore`, `.dockerignore`,
  `.gitattributes`, and `.env.example` must not disappear during a file copy.
- Create an empty GitHub repository with no generated README/license, then
  connect it using your actual URL:

```text
git remote add origin https://github.com/YOUR_USERNAME/project-phantom.git
git push -u origin main
```

Do not add another `origin` if one already exists. Wait for CI and Docker checks,
then test full real mode and browser permissions on representative Windows and
Linux machines before describing the release as validated on those platforms.

## Troubleshooting

- **`docker` not found:** install Docker Desktop on Windows, or Engine/Compose
  on Linux, then reopen the terminal. A source checkout does not install Docker.
- **Cannot connect to daemon:** start Docker Desktop/the Engine service. On
  Windows verify WSL 2 and Linux-container mode.
- **Port occupied:** set `PHANTOM_PORT=8001` in `.env` and recreate the app.
- **Browser cannot use sensors:** open the loopback URL, grant only consented
  camera/microphone permissions, and close applications holding the device.
  Docker does not need device passthrough because capture happens in the browser.
- **Container exits with code 137/OOM:** increase Docker/WSL memory, close other
  applications, or use demo mode. Do not replace failed real outputs with demo.
- **Unhealthy on first real start:** view logs and allow download/warm-up time.
  Check internet access, free disk, and RAM. HTTP `healthy` proves liveness only;
  `/api/health` and the real smoke check must report all components ready.
- **A backend is unavailable:** stop the running app before launching a separate
  model checker, to avoid loading a second full stack into RAM:

```text
docker compose -f docker-compose.yml -f compose.realtime.yaml down
docker compose -f docker-compose.yml -f compose.realtime.yaml run --rm --no-deps phantom python scripts/check_realtime_models.py
```

  Inspect the JSON for all six `readiness: ready` values, not just the process
  exit code. Restart with the normal real-mode `up -d` command afterward.
- **No space left:** inspect `docker system df`. Do not blindly run volume/system
  prune commands; they can delete unrelated containers, images, or caches.
- **Rebuild after changes:** use `up --build -d` for the selected profile. If
  updating pinned ML versions, rebuild the real image and rerun the dependency
  and full-real acceptance checks. Direct dependency pins are not a complete
  transitive lock; use a commit-specific prebuilt image for a fixed installation.
