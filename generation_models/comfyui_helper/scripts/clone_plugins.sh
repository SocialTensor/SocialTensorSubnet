if [ "$(id -u)" -ne 0 ]; then
  # If not root, use sudo
  sudo curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/download/v0.6.0/pget_linux_x86_64"
  sudo chmod +x /usr/local/bin/pget
  sudo chown $(whoami) /run/pget.pid
else
  # If root, run the commands directly
  curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/download/v0.6.0/pget_linux_x86_64"
  chmod +x /usr/local/bin/pget
fi
cd generation_models/comfyui_helper/ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager
git clone https://github.com/ZHO-ZHO-ZHO/ComfyUI-BRIA_AI-RMBG
git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale --recursive
git clone https://github.com/jags111/efficiency-nodes-comfyui && cd efficiency-nodes-comfyui && git checkout a5422d6599971881b5fc80d0496ca3a8b6108267 && cd ..
git clone https://github.com/WASasquatch/was-node-suite-comfyui

git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus && git checkout 4e898fe
git clone https://github.com/Fannovel16/comfyui_controlnet_aux && git checkout 6d6f63c
git clone https://github.com/cubiq/ComfyUI_InstantID
pip install insightface
git clone https://github.com/cubiq/ComfyUI_essentials && cd ComfyUI_essentials && git checkout bd9b89b7c924302e14bb353b87c3373af447bf55 && cd ..

cd was-node-suite-comfyui
pip install -r requirements.txt
pip install onnxruntime-gpu
