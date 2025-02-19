virtualenv ./venv --python=python3
. ./venv/bin/activate

# torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# mmdetection
pip install openmim
mim install mmengine
mim install "mmcv>=2.0.0" # when this does not work, install mmcv from source https://mmcv.readthedocs.io/en/latest/get_started/build.html#


pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.

mim install mmdet

# mmdetection verification
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu

# from mmdet.apis import init_detector, inference_detector
# config_file = 'rtmdet_tiny_8xb32-300e_coco.py'
# checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
# model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
# inference_detector(model, 'demo/demo.jpg')

# For PyQt Error: Could not load the Qt platform plugin "xcb"
pip install opencv-python-headless PyQt5
# sudo apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev libxcb-xinerama0

pip install piq # for image quality metrics
