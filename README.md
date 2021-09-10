# Data analysis
- Document here the project: Cryptotradingindicator
- Description: 
This project was a two weeks project done by: Tim Jablonski, Frieda Tims-Jansma, Ivan Fernandes and Jhordy Mora. The objective of this was to try to predict the price of Bitcoin on a 4H timeframe using all the knowledge acquired in the Data Science and machine learning Bootcamp in LeWagon Berlin. Using different tools like web scraping and pandas for data recollecting and preprocessing, github, scikit learn and tensorflow for the design of the model using Time-Series (LSTM) and Docker, Heroku and GCP for the Front-end.
- Data Source: Gemini Exchange
- Type of analysis: Time-Series(LSTM)


# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for cryptotrading-indicator in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/cryptotradingindicator`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "cryptotrading-indicator"
git remote add origin git@github.com:{group}/cryptotradingindicator.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
cryptotradingindicator-run
```

# Install

Go to `https://github.com/{group}/cryptotradingindicator` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/cryptotradingindicator.git
cd cryptotradingindicator
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
cryptotrading-indicator-run
```
