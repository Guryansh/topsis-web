# build_files.sh
apt-get update
apt-get install -y python3-pip python3-venv
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt

# Run Django collectstatic to gather static files
python manage.py collectstatic

# Move the collected static files to the expected directory
mkdir -p staticfiles_build
mv static staticfiles_build/static