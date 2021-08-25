# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* cryptotrading-indicator/*.py

black:
	@black scripts/* cryptotrading-indicator/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr cryptotrading-indicator-*.dist-info
	@rm -fr cryptotrading-indicator.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

streamlit:
	-@streamlit run crypto_web.py

heroku_login:
	-@heroku login

heroku_create_app:
	-@heroku create ${APP_NAME}

deploy_heroku:
	-@git push heroku master
	-@heroku ps:scale web=1


## CREATE GOOGLE CLOUD STORAGE BUCKET
# project id - replace with your GCP project id
PROJECT_ID=crypto-indicator

# bucket name - replace with your GCP bucket name
BUCKET_NAME=crypto-indicator

# choose your region from https://cloud.google.com/storage/docs/locations#available_locations
REGION=europe-west1

set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}


## UPLOAD FILE TO GOOGLE CLOUD STORAGE
# path to the file to upload to GCP (the path to the file should be absolute or should match the directory where the make command is ran)
# replace with your local path to the `train_1k.csv` and make sure to put the path between quotes
LOCAL_PATH="/home/ivanfernandes/code/timjab92/cryptotradingindicator/data/BTCUSD_2011-12-31_to_2021-08-23_4hours_Clean.csv"

# bucket directory in which to store the uploaded file (`data` is an arbitrary name that we choose to use)
BUCKET_FOLDER=data

# name for the uploaded file inside of the bucket (we choose not to rename the file that we upload)
BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})

upload_data:
    # @gsutil cp file gs://my-bucket-name/data/file_name.csv
	@gsutil cp ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}



# BUCKET_TRAINING_FOLDER =training

# PYTHON_VERSION=3.7
# FRAMEWORK=scikit-learn
# RUNTIME_VERSION=1.15

# PACKAGE_NAME=cryptotradingindicator
# FILENAME=trainer

# JOB_NAME=crypto_$(shell date +'%Y%m%d_%H%M%S')

# ## SETUP GOOGLE AI PLATFORM
# gcloud ai-platform jobs submit training ${JOB_NAME} \
#   --job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER}  \
#   --package-path ${PACKAGE_NAME} \
#   --module-name ${PACKAGE_NAME}.${FILENAME} \
#   --python-version=${PYTHON_VERSION} \
#   --runtime-version=${RUNTIME_VERSION} \
#   --region ${REGION} \
#   --stream-logs