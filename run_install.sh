echo "Intasll wget..."
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install wget --force-yes
echo "Installing python"
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
echo "Creating Python virtualenv..."
conda create -n test python=3.9
conda activate test
echo "Downloading VGG16_bn files..."
mkdir backbone_model
curl -o vgg16_bn-6c64b313.pth https://download.pytorch.org/models/vgg16_bn-6c64b313.pth
mv vgg16_bn-6c64b313.pth backbone_model/
echo "Installing Python dependencies..."
pip install -r requirements.txt