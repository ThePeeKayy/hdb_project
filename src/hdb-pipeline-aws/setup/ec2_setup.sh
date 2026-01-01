set -e

echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y


echo "Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    python3-dev \
    libgomp1


echo "Installing Miniconda..."
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh -b -p /home/ubuntu/miniconda3
rm Miniconda3-latest-Linux-aarch64.sh


/home/ubuntu/miniconda3/bin/conda init bash
source ~/.bashrc


echo "Creating Python environment..."
/home/ubuntu/miniconda3/bin/conda create -n hdb-env python=3.11 -y
source /home/ubuntu/miniconda3/bin/activate hdb-env


echo "Installing Python packages..."
pip install --upgrade pip


pip install \
    pandas==2.1.4 \
    numpy==1.26.2 \
    pyarrow==14.0.1 \
    boto3==1.34.0


pip install \
    darts==0.27.2 \
    torch==2.1.1 \
    pytorch-lightning==2.1.2 \
    scikit-learn==1.3.2


pip install \
    requests==2.31.0 \
    python-dateutil==2.8.2

echo "Python packages installed successfully!"


echo "Creating directory structure..."
mkdir -p /home/ubuntu/pipeline
mkdir -p /home/ubuntu/logs


echo "Installing AWS CLI..."
cd /tmp
curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip


echo "AWS CLI installed. Run 'aws configure' to set up credentials."


echo "Setting up cron jobs..."


(crontab -l 2>/dev/null; echo "
(crontab -l 2>/dev/null; echo "
(crontab -l 2>/dev/null; echo "0 18 * * * /home/ubuntu/pipeline/daily_pipeline.sh >> /home/ubuntu/logs/cron.log 2>&1") | crontab -
(crontab -l 2>/dev/null; echo "
(crontab -l 2>/dev/null; echo "0 17 * * * /home/ubuntu/pipeline/weekly_retrain.sh >> /home/ubuntu/logs/cron.log 2>&1") | crontab -

echo "Cron jobs configured!"


chmod +x /home/ubuntu/pipeline/*.sh

aws s3 ls || echo "Note: Configure AWS credentials with 'aws configure'"

echo "Cron schedule:"
echo "  - Daily pipeline: 2 AM SGT (6 PM UTC)"
echo "  - Weekly retrain: 1 AM SGT Sunday (5 PM UTC Saturday)"
echo ""