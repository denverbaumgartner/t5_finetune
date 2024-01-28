# curl https://pyenv.run | bash
# pyenv install 3.11.6
# pyenv global 3.11.6

python3 -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
# python -m ipykernel install --user --name=myenvkernel --display-name=sql-dataset-tuning