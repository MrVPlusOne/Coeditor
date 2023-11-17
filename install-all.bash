# Assuming on ubuntu, run commands to install all depedencies

curl -sSL https://install.python-poetry.org | python3 -
poetry completions bash >> ~/.bash_completion

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.11

poetry install