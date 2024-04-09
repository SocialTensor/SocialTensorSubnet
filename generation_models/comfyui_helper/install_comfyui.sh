python -m venv comfyui
. comfyui/bin/activate
git submodule update --init --recursive
pip install -r generation_models/comfyui_helper/ComfyUI/requirements.txt
bash generation_models/comfyui_helper/scripts/clone_plugins.sh
deactivate