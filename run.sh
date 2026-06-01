#!/usr/bin/env bash
# =============================================================================
# SeedVR2 — one-command setup + launch (single GPU: RTX 5090 32GB or A6000 48GB)
# Installs deps, clones the repo, writes webui.py, downloads the model, launches.
#
# Usage on a fresh Vast.ai instance (CUDA 12.8 PyTorch template, -p 7860:7860):
#   curl -fsSL https://raw.githubusercontent.com/<you>/<repo>/main/run.sh | bash
#
# Optional env overrides (prepend before the command):
#   MODEL=...      DiT model file (default: 7B fp16, best quality)
#   ATTENTION=...  attention backend (default: flash_attn_2; 5090 -> sageattn_3)
#   SKIP_LAUNCH=1  set up everything but don't start the server
# =============================================================================
set -euo pipefail

REPO_URL="https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
REPO_DIR="ComfyUI-SeedVR2_VideoUpscaler"
MODEL="${MODEL:-seedvr2_ema_7b_fp16.safetensors}"
ATTENTION="${ATTENTION:-flash_attn_2}"

blue() { printf "\n\033[1;34m==> %s\033[0m\n" "$*"; }

blue "Installing system deps (ffmpeg, git)"
apt-get update && apt-get install -y ffmpeg git

blue "GPU sanity check (must pass before anything else)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '|', torch.cuda.get_device_name(0))" || {
  echo "!! PyTorch/CUDA not usable. Redeploy with a CUDA 12.8 PyTorch template."; exit 1;
}

blue "Cloning SeedVR2"
[ -d "$REPO_DIR" ] || git clone "$REPO_URL"
cd "$REPO_DIR"

blue "Installing Python deps"
pip install -r requirements.txt
pip install gradio "huggingface_hub[cli]"
# flash-attn: avoids a transformers KeyError('flash_attn') and powers flash_attn_2.
pip install flash-attn --no-build-isolation || echo "!! flash-attn build failed; use ATTENTION=sdpa if launch crashes."
# SageAttention: only used by the 5090-only sageattn_3 backend; ignore failures.
pip install sageattention || true

blue "Writing webui.py"
cat > webui.py <<'PYEOF'
#!/usr/bin/env python3
"""
Node-free Gradio WebUI wrapping numz SeedVR2 inference_cli.py.

Defaults tuned for a SINGLE GPU (one RTX 5090 32 GB, or an A6000 48 GB), fully
speed-optimised: model kept resident on the GPU (no CPU offload), fast attention
backend, torch.compile on DiT + VAE, large 4n+1 batches, and streaming
(--chunk_size) so long videos stay within RAM.

Multi-GPU is fully supported and just one field away: set the CUDA devices box to
'0,1' (or '0,1,2,3').
"""
import os
import shlex
import sys
import subprocess

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gradio as gr

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
CLI = os.path.join(REPO_DIR, "inference_cli.py")

DEFAULT_MODEL = os.environ.get("MODEL", "seedvr2_ema_7b_fp16.safetensors")
DEFAULT_ATTENTION = os.environ.get("ATTENTION", "flash_attn_2")

DIT_MODELS = [
    "seedvr2_ema_7b_fp16.safetensors",
    "seedvr2_ema_7b_sharp_fp16.safetensors",
    "seedvr2_ema_7b-Q8_0.gguf",
    "seedvr2_ema_3b_fp16.safetensors",
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "seedvr2_ema_3b-Q8_0.gguf",
]
if DEFAULT_MODEL not in DIT_MODELS:
    DIT_MODELS.insert(0, DEFAULT_MODEL)

ATTENTION_MODES = ["flash_attn_2", "sageattn_3", "sageattn_2", "sdpa"]
BATCH_CHOICES = [1, 5, 9, 13, 17, 21, 25, 33, 41, 49, 65, 81, 97]
VRAM_STRATEGIES = [
    "Resident (fastest - model stays on the GPU)",
    "Resident + cached (CPU offload + cache between chunks; very long clips)",
    "CPU offload + block swap (low-VRAM fallback)",
]


def build_command(input_path, output_dir, cuda_devices, dit_model, attention_mode,
                  resolution, max_resolution, batch_size, uniform_batch,
                  temporal_overlap, prepend_frames, chunk_size, vram_strategy,
                  blocks_to_swap, swap_io, compile_dit, compile_vae, vae_tiling,
                  vae_tile_size, vae_tile_overlap, color_correction, video_backend,
                  ten_bit, extra_args):
    cmd = [sys.executable, "-u", CLI, input_path]
    if output_dir:
        cmd += ["--output", output_dir]
    cmd += ["--dit_model", dit_model]
    cmd += ["--cuda_device", (cuda_devices.strip() or "0")]
    cmd += ["--attention_mode", attention_mode]
    cmd += ["--resolution", str(int(resolution))]
    if int(max_resolution) > 0:
        cmd += ["--max_resolution", str(int(max_resolution))]
    cmd += ["--batch_size", str(int(batch_size))]
    if uniform_batch:
        cmd += ["--uniform_batch_size"]
    if int(temporal_overlap) > 0:
        cmd += ["--temporal_overlap", str(int(temporal_overlap))]
    if int(prepend_frames) > 0:
        cmd += ["--prepend_frames", str(int(prepend_frames))]
    if int(chunk_size) > 0:
        cmd += ["--chunk_size", str(int(chunk_size))]
    cmd += ["--color_correction", color_correction]
    if compile_dit:
        cmd += ["--compile_dit"]
    if compile_vae:
        cmd += ["--compile_vae"]
    if vram_strategy.startswith("Resident + cached"):
        cmd += ["--dit_offload_device", "cpu", "--vae_offload_device", "cpu",
                "--cache_dit", "--cache_vae"]
    elif vram_strategy.startswith("CPU offload"):
        cmd += ["--dit_offload_device", "cpu", "--vae_offload_device", "cpu"]
        if int(blocks_to_swap) > 0:
            cmd += ["--blocks_to_swap", str(int(blocks_to_swap))]
            if swap_io:
                cmd += ["--swap_io_components"]
    else:
        cmd += ["--dit_offload_device", "none", "--vae_offload_device", "none"]
    if vae_tiling:
        cmd += ["--vae_encode_tiled", "--vae_decode_tiled",
                "--vae_encode_tile_size", str(int(vae_tile_size)),
                "--vae_encode_tile_overlap", str(int(vae_tile_overlap)),
                "--vae_decode_tile_size", str(int(vae_tile_size)),
                "--vae_decode_tile_overlap", str(int(vae_tile_overlap))]
    cmd += ["--video_backend", video_backend]
    if ten_bit and video_backend == "ffmpeg":
        cmd += ["--10bit"]
    if extra_args and extra_args.strip():
        cmd += shlex.split(extra_args)
    return cmd


def _newest_new_file(out_dir, before):
    if not os.path.isdir(out_dir):
        return None
    after = [os.path.join(out_dir, f) for f in os.listdir(out_dir)
             if os.path.isfile(os.path.join(out_dir, f))]
    new_files = [f for f in after if f not in before]
    candidates = new_files if new_files else after
    return max(candidates, key=os.path.getmtime) if candidates else None


def run_upscale(video, server_path, output_dir, cuda_devices, dit_model,
                attention_mode, resolution, max_resolution, batch_size,
                uniform_batch, temporal_overlap, prepend_frames, chunk_size,
                vram_strategy, blocks_to_swap, swap_io, compile_dit, compile_vae,
                vae_tiling, vae_tile_size, vae_tile_overlap, color_correction,
                video_backend, ten_bit, extra_args):
    if server_path and server_path.strip():
        p = server_path.strip()
        video = p if os.path.isabs(p) else os.path.join(REPO_DIR, p)
        if not os.path.isfile(video):
            yield "No file found at: %s" % video, None, None
            return
    if not video:
        yield "Please upload a video OR type a path to a file on the instance.", None, None
        return
    out_dir = output_dir if os.path.isabs(output_dir) else os.path.join(REPO_DIR, output_dir)
    os.makedirs(out_dir, exist_ok=True)
    sep = "-" * 60
    before = ({os.path.join(out_dir, f) for f in os.listdir(out_dir)}
              if os.path.isdir(out_dir) else set())
    cmd = build_command(
        video, output_dir, cuda_devices, dit_model, attention_mode, resolution,
        max_resolution, batch_size, uniform_batch, temporal_overlap, prepend_frames,
        chunk_size, vram_strategy, blocks_to_swap, swap_io, compile_dit, compile_vae,
        vae_tiling, vae_tile_size, vae_tile_overlap, color_correction, video_backend,
        ten_bit, extra_args)
    log = "Running on GPU(s): %s  (watch with: watch -n1 nvidia-smi)\n" % (
        cuda_devices.strip() or "0")
    log += "Command:\n" + " ".join(cmd) + "\n" + sep + "\n"
    yield log, None, None
    proc = subprocess.Popen(cmd, cwd=REPO_DIR, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        log += line
        yield log, None, None
    proc.wait()
    if proc.returncode != 0:
        log += "\n%s\nFailed (exit %d). See log above.\n" % (sep, proc.returncode)
        yield log, None, None
        return
    result = _newest_new_file(out_dir, before)
    log += "\n%s\nDone!" % sep
    if result:
        log += "\nOutput: %s" % result
    yield log, result, result


with gr.Blocks(title="SeedVR2 Upscaler") as demo:
    gr.Markdown("## SeedVR2 Video Upscaler - single GPU (5090 / A6000), multi-GPU optional")
    with gr.Row():
        with gr.Column():
            video = gr.Video(label="Input video (or image)")
            server_path = gr.Textbox(
                label="...or path to a file already on the instance (skips upload)",
                placeholder="input/myvideo.mp4")
            output_dir = gr.Textbox(label="Output folder", value="output/")
            cuda_devices = gr.Textbox(
                value="0",
                label="CUDA devices ('0' = single GPU; '0,1' = use both GPUs)")
            dit_model = gr.Dropdown(DIT_MODELS, value=DEFAULT_MODEL,
                                    label="Model (7B fp16 = best quality; 3B fp8 = fastest, 5090 only)")
            attention_mode = gr.Dropdown(ATTENTION_MODES, value=DEFAULT_ATTENTION,
                                         label="Attention (flash_attn_2 = A6000+5090; sageattn_3 = 5090 only, faster)")
            color_correction = gr.Dropdown(
                ["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"],
                value="lab", label="Color correction")
            with gr.Row():
                resolution = gr.Slider(480, 2160, value=1080, step=8,
                                       label="Target short-side resolution")
                max_resolution = gr.Slider(0, 3840, value=0, step=8,
                                           label="Max edge (0 = no limit)")
            batch_size = gr.Dropdown(BATCH_CHOICES, value=33,
                                     label="Batch size (4n+1; push higher for speed, lower if OOM)")
            uniform_batch = gr.Checkbox(value=True, label="Uniform batch size")
            extra_args = gr.Textbox(
                value="",
                label="Extra CLI flags (advanced)",
                placeholder="verify names via: python inference_cli.py --help")
        with gr.Column():
            gr.Markdown("### Speed (attention + compile)")
            compile_dit = gr.Checkbox(value=True, label="torch.compile DiT (~20-40% faster)")
            compile_vae = gr.Checkbox(value=True, label="torch.compile VAE (~15-25% faster)")
            gr.Markdown("### VRAM (32+ GB: keep model resident)")
            vram_strategy = gr.Dropdown(VRAM_STRATEGIES, value=VRAM_STRATEGIES[0],
                                        label="VRAM strategy")
            blocks_to_swap = gr.Slider(0, 36, value=0, step=1,
                                       label="Blocks to swap (only for CPU-offload fallback)")
            swap_io = gr.Checkbox(value=False, label="Swap I/O components (only if offloading)")
            vae_tiling = gr.Checkbox(value=False, label="VAE tiling (leave OFF on 32+ GB)")
            vae_tile_size = gr.Slider(256, 1024, value=1024, step=64,
                                      label="VAE tile size (only if tiling)")
            vae_tile_overlap = gr.Slider(0, 256, value=64, step=16,
                                         label="VAE tile overlap (only if tiling)")
            gr.Markdown("### Long video (streaming + seam blending)")
            chunk_size = gr.Slider(0, 1024, value=200, step=1,
                                   label="Chunk size (frames streamed at once; 0 = load all)")
            temporal_overlap = gr.Slider(0, 8, value=3, step=1,
                                         label="Temporal overlap (blends batch/GPU seams)")
            prepend_frames = gr.Slider(0, 8, value=4, step=1,
                                       label="Prepend frames (reduces start-of-clip artifacts)")
            gr.Markdown("### Output")
            video_backend = gr.Dropdown(["opencv", "ffmpeg"], value="ffmpeg",
                                        label="Video backend")
            ten_bit = gr.Checkbox(value=False, label="10-bit x265 (needs ffmpeg)")
    run_btn = gr.Button("Upscale", variant="primary")
    with gr.Row():
        output_video = gr.Video(label="Result (preview)")
        output_file = gr.File(label="Download upscaled file")
    output_log = gr.Textbox(label="Log", lines=20, max_lines=20, autoscroll=True)
    run_btn.click(
        run_upscale,
        inputs=[video, server_path, output_dir, cuda_devices, dit_model,
                attention_mode, resolution, max_resolution, batch_size,
                uniform_batch, temporal_overlap, prepend_frames, chunk_size,
                vram_strategy, blocks_to_swap, swap_io, compile_dit, compile_vae,
                vae_tiling, vae_tile_size, vae_tile_overlap, color_correction,
                video_backend, ten_bit, extra_args],
        outputs=[output_log, output_video, output_file],
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
PYEOF

blue "Downloading model: $MODEL (this is the big download)"
hf download numz/SeedVR2_comfyUI "$MODEL" --local-dir ./models/SEEDVR2

if [ "${SKIP_LAUNCH:-0}" = "1" ]; then
  blue "Setup complete. Launch later with:  cd $REPO_DIR && MODEL=$MODEL ATTENTION=$ATTENTION python webui.py"
  exit 0
fi

blue "Launching WebUI on :7860  (open your instance's external IP:port for 7860)"
MODEL="$MODEL" ATTENTION="$ATTENTION" python webui.py
