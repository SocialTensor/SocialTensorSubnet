curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/download/v0.6.0/pget_linux_x86_64" && chmod +x /usr/local/bin/pget
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

cd generation_models/comfyui_helper/ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager
git clone https://github.com/ZHO-ZHO-ZHO/ComfyUI-BRIA_AI-RMBG
git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale --recursive
git clone https://github.com/jags111/efficiency-nodes-comfyui
git clone https://github.com/WASasquatch/was-node-suite-comfyui

git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus && git checkout 4e898fe
git clone https://github.com/Fannovel16/comfyui_controlnet_aux && git checkout 6d6f63c
git clone https://github.com/cubiq/ComfyUI_InstantID
pip install insightface
git clone https://github.com/cubiq/ComfyUI_essentials && git checkout c9236fe

cd was-node-suite-comfyui
pip install -r requirements.txt