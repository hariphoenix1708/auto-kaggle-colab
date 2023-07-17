#!/bin/bash

# - Dependencies - #
env TF_CPP_MIN_LOG_LEVEL=1
apt -y update -qq
apt -y install -qq aria2
wget http://launchpadlibrarian.net/367274644/libgoogle-perftools-dev_2.5-2.2ubuntu3_amd64.deb
wget https://launchpad.net/ubuntu/+source/google-perftools/2.5-2.2ubuntu3/+build/14795286/+files/google-perftools_2.5-2.2ubuntu3_all.deb
wget https://launchpad.net/ubuntu/+source/google-perftools/2.5-2.2ubuntu3/+build/14795286/+files/libtcmalloc-minimal4_2.5-2.2ubuntu3_amd64.deb
wget https://launchpad.net/ubuntu/+source/google-perftools/2.5-2.2ubuntu3/+build/14795286/+files/libgoogle-perftools4_2.5-2.2ubuntu3_amd64.deb
apt -y install -qq libunwind8-dev
dpkg -i *.deb
env LD_PRELOAD=libtcmalloc.so
rm *.deb
pip install --upgrade setuptools
python -m pip cache purge

# pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

!apt-get -y install -qq aria2
!curl -Lo microsoftexcel.zip https://huggingface.co/nolanaatama/colab/resolve/main/microsoftexcel140.zip
!unzip /content/microsoftexcel.zip

# --- MODELS, LORAS, VAE --- #
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/phoenix-1708/DR_NED/resolve/main/malklat-woman-1611-malklat.ckpt -d /content/microsoftexcel/models/Stable-diffusion -o malklat-woman-1611-malklat.ckpt
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/phoenix-1708/DR_NED/resolve/main/iswmnn-woman-1886-iswmnn.ckpt -d /content/microsoftexcel/models/Stable-diffusion -o iswmnn-woman-1886-iswmnn.ckpt
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/phoenix-1708/DR_NED/resolve/main/DR_NED_120623.safetensors -d /content/microsoftexcel/models/Stable-diffusion -o DR_NED_120623.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors -d /content/microsoftexcel/models/VAE -o vae-ft-mse-840000-ema-pruned.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/vae/kl-f8-anime2.ckpt -d /content/microsoftexcel/models/VAE -o kl-f8-anime2.ckpt

# --- UPSCALERS --- #
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Zabin/Resizers/resolve/main/4x-UltraSharp.pth -d /content/microsoftexcel/models/ESRGAN -o 4x-UltraSharp.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Zabin/Resizers/resolve/main/4x_foolhardy_Remacri.pth -d /content/microsoftexcel/models/ESRGAN -o 4x_foolhardy_Remacri.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/gemasai/4x_NMKD-Superscale-SP_178000_G/resolve/main/4x_NMKD-Superscale-SP_178000_G.pth -d /content/microsoftexcel/models/ESRGAN -o 4x_NMKD-Superscale-SP_178000_G.pth

# --- EMBEDDINGS --- #
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/negative/resolve/main/EasyNegative.pt -d /content/microsoftexcel/embeddings/Negative -o EasyNegative.pt
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/gsdf/Counterfeit-V3.0/resolve/main/embedding/EasyNegativeV2.safetensors -d /content/microsoftexcel/embeddings/Negative -o EasyNegativeV2.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/negative/resolve/main/bad-hands-5.pt -d /content/microsoftexcel/embeddings/Negative -o bad-hands-5.pt
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/negative/resolve/main/bad_prompt_version2.pt -d /content/microsoftexcel/embeddings/Negative -o bad_prompt_version2.pt
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/negative/resolve/main/ng_deepnegative_v1_75t.pt -d /content/microsoftexcel/embeddings/Negative -o ng_deepnegative_v1_75t.pt
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ffxvs/negative-prompts-pack/resolve/main/bad-artist.pt -d /content/microsoftexcel/embeddings/Negative -o bad-artist.pt
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ffxvs/negative-prompts-pack/resolve/main/badhandv4.pt -d /content/microsoftexcel/embeddings/Negative -o badhandv4.pt
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ffxvs/negative-prompts-pack/resolve/main/verybadimagenegative_v1.3.pt -d /content/microsoftexcel/embeddings/Negative -o verybadimagenegative_v1.3.pt
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://civitai.com/api/download/models/60938 -d /content/microsoftexcel/embeddings/Negative -o negative_hand-neg.pt

# --- EXTENSIONS --- #
git clone https://github.com/zanllp/sd-webui-infinite-image-browsing /content/microsoftexcel/extensions/sd-webui-infinite-image-browsing
git clone https://github.com/KohakuBlueleaf/a1111-sd-webui-lycoris /content/microsoftexcel/extensions/a1111-sd-webui-lycoris
git clone https://github.com/etherealxx/batchlinks-webui /content/microsoftexcel/extensions/batchlinks-webui
git clone https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111 /content/microsoftexcel/extensions/multidiffusion-upscaler-for-automatic1111
git clone https://github.com/mcmonkeyprojects/sd-dynamic-thresholding /content/microsoftexcel/extensions/sd-dynamic-thresholding
git clone https://github.com/DominikDoom/a1111-sd-webui-tagcomplete /content/microsoftexcel/extensions/a1111-sd-webui-tagcomplete
git clone https://github.com/aka7774/sd_filer /content/microsoftexcel/extensions/sd_filer
git clone https://github.com/hako-mikan/sd-webui-supermerger /content/microsoftexcel/extensions/sd-webui-supermerger
git clone https://github.com/Bing-su/adetailer /content/microsoftexcel/extensions/adetailer
git clone https://github.com/deforum-art/deforum-for-automatic1111-webui /content/microsoftexcel/extensions/deforum-for-automatic1111-webui
git clone https://github.com/thomasasfk/sd-webui-aspect-ratio-helper /content/microsoftexcel/extensions/sd-webui-aspect-ratio-helper
git clone https://github.com/Elldreth/loopback_scaler /content/microsoftexcel/extensions/loopback_scaler
git clone https://github.com/hariphoenix1708/fswap /content/microsoftexcel/extensions/roop

# ControlNet, if you need it → ( ↓ Sellect commented lines below and press "Ctrl + /" to uncomment all at once ↓ )

git clone https://github.com/Mikubill/sd-webui-controlnet /content/microsoftexcel/extensions/sd-webui-controlnet
# aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11e_sd15_ip2p_fp16.safetensors -d /content/microsoftexcel/extensions/sd-webui-controlnet/models -o control_v11e_sd15_ip2p_fp16.safetensors
# aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11e_sd15_shuffle_fp16.safetensors -d /content/microsoftexcel/extensions/sd-webui-controlnet/models -o control_v11e_sd15_shuffle_fp16.safetensors
# aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny_fp16.safetensors -d /content/microsoftexcel/extensions/sd-webui-controlnet/models -o control_v11p_sd15_canny_fp16.safetensors
# aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_depth_fp16.safetensors -d /content/microsoftexcel/extensions/sd-webui-controlnet/models -o control_v11p_sd15_depth_fp16.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors -d /content/microsoftexcel/extensions/sd-webui-controlnet/models -o control_v11p_sd15_inpaint_fp16.safetensors
# aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart_fp16.safetensors -d /content/microsoftexcel/extensions/sd-webui-controlnet/models -o control_v11p_sd15_lineart_fp16.safetensors
# aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_mlsd_fp16.safetensors -d /content/microsoftexcel/extensions/sd-webui-controlnet/models -o control_v11p_sd15_mlsd_fp16.safetensors
# aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae_fp16.safetensors -d /content/microsoftexcel/extensions/sd-webui-controlnet/models -o control_v11p_sd15_normalbae_fp16.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose_fp16.safetensors -d /content/microsoftexcel/extensions/sd-webui-controlnet/models -o control_v11p_sd15_openpose_fp16.safetensors
# aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble_fp16.safetensors -d /content/microsoftexcel/extensions/sd-webui-controlnet/models -o control_v11p_sd15_scribble_fp16.safetensors
# aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_seg_fp16.safetensors -d /content/microsoftexcel/extensions/sd-webui-controlnet/models -o control_v11p_sd15_seg_fp16.safetensors
# aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge_fp16.safetensors -d /content/microsoftexcel/extensions/sd-webui-controlnet/models -o control_v11p_sd15_softedge_fp16.safetensors
# aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15s2_lineart_anime_fp16.safetensors -d /content/microsoftexcel/extensions/sd-webui-controlnet/models -o control_v11p_sd15s2_lineart_anime_fp16.safetensors
# aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11u_sd15_tile_fp16.safetensors -d /content/microsoftexcel/extensions/sd-webui-controlnet/models -o control_v11u_sd15_tile_fp16.safetensors
# aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_style_sd14v1.pth -d /content/microsoftexcel/extensions/sd-webui-controlnet/models -o t2iadapter_style_sd14v1.pth
# aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_sketch_sd14v1.pth -d /content/microsoftexcel/extensions/sd-webui-controlnet/models -o t2iadapter_sketch_sd14v1.pth
# aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_seg_sd14v1.pth -d /content/microsoftexcel/extensions/sd-webui-controlnet/models -o t2iadapter_seg_sd14v1.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_openpose_sd14v1.pth -d /content/microsoftexcel/extensions/sd-webui-controlnet/models -o t2iadapter_openpose_sd14v1.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_keypose_sd14v1.pth -d /content/microsoftexcel/extensions/sd-webui-controlnet/models -o t2iadapter_keypose_sd14v1.pth
# aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_color_sd14v1.pth -d /content/microsoftexcel/extensions/sd-webui-controlnet/models -o t2iadapter_color_sd14v1.pth
# aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_canny_sd15v2.pth -d /content/microsoftexcel/extensions/sd-webui-controlnet/models -o t2iadapter_canny_sd15v2.pth
# aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_depth_sd15v2.pth -d /content/microsoftexcel/extensions/sd-webui-controlnet/models -o t2iadapter_depth_sd15v2.pth
# aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_sketch_sd15v2.pth -d /content/microsoftexcel/extensions/sd-webui-controlnet/models -o t2iadapter_sketch_sd15v2.pth
# aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_zoedepth_sd15v1.pth -d /content/microsoftexcel/extensions/sd-webui-controlnet/models -o t2iadapter_zoedepth_sd15v1.pth
