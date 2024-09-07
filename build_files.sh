# build_files.sh
apt-get update
apt-get install -y python3-pip python3-venv
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt

# Run Django collectstatic to gather static files
python manage.py collectstatic --no-post-process
python manage.py makemigrations
python manage.py migrate
# Move the collected static files to the expected directory
mkdir -p staticfiles_build
cp -r static staticfiles_build/static