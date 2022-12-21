# RoqueAlgorithmsSimilarityServer
A Flask server for computing the similarity between articles

In `HPC-setup.md` are steps for setting up/ starting up the workspace.

## Installation
1. Clone the repository:
```commandline
git clone https://github.com/uashogeschoolutrecht/RoqueAlgorithmsSimilarityServer
```
2. get into the directory
```commandline
cd RoqueAlgorithmsSimilarityServer
```
3. create a virtual python environment in that directory to install all packages 
```commandline
python3 -m venv venv
```
4. activate that environment
```commandline
source venv/bin/activate
```
5. install packages from requirements.txt with pip
```commandline
pip install -r requirements.txt
```

## Run python
To run 
```commandline
python3 src/similarity_server.py
```
## Run Docker
Make sure `Docker` is running
```
Docker compose up
```