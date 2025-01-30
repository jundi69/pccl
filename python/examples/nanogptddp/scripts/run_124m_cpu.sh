MAIN_DIR=$(pwd)/..

cd "$MAIN_DIR" || exit
python train.py --config_path "default_config_cpu.json"