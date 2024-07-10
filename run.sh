export PATH=/usr/local/cuda/bin:$PATH
echo $OPENAI_API_KEY
cd GLIP
# python setup.py clean --all build develop --user
cd ..
CUDA_VISIBLE_DEVICES=2,3 CONFIG_NAMES=covr_config python main_batch.py
