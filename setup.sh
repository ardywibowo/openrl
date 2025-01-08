pip install -r requirements.txt
pip install flash-attn --no-build-isolation

apt-get update -y
apt-get install -y cuda-libraries-dev-12-4
apt install libssl-dev

chmod a+x scripts/download_and_prepare_dataset.sh
./scripts/download_and_prepare_dataset.sh

git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
git checkout v0.15.4
DS_BUILD_OPS=1 pip install . --global-option="build_ext" --global-option="-j8"
ds_report
