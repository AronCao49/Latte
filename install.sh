pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu118.html
pip install --default-timeout=100 spconv-cu118
pip install --default-timeout=100 timm
pip install --default-timeout=100 openpyxl

# install this repo
pip install -ve .

pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0