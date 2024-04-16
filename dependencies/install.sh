# Installing dependencies for the project
echo "Installing dependencies for the project"
sudo apt install patchelf libglew-dev libosmesa6-dev libgl1-mesa-glx libglfw3 libopengl0 -y

# Creating conda environment
echo "\nCreating conda environment"
conda env create -f ./dependencies/environment.yml
eval "$(conda shell.bash hook)"
conda activate continuous

# Installing setuptools and importlib-metadata for gym v0.21.0 
echo "\nInstalling setuptools and importlib-metadata for gym v0.21.0"
pip install setuptools==59.5.0 importlib-metadata==4.13.0

# Installing other dependencies
echo "\nInstalling other dependencies"
pip install -r ./dependencies/requirements.txt

# Installing torch and signatory
echo "\nInstalling torch and signatory"
pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0
pip install signatory==1.2.6.1.9.0
