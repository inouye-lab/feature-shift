import sys
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import urllib.request
import zipfile


print(sys.version)

# Gas
def fetch_gas_data():
    final_filename = data_path/'gas-sensor-array-drift.dat'
    if not Path.exists(final_filename):
        temp_path = Path('temp-data')
        if not Path.exists(temp_path):
            Path.mkdir(temp_path)
        uci_gas_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00362/HT_Sensor_UCIsubmission.zip'
        print('Downloading Gas Dataset')
        response = urllib.request.urlretrieve(uci_gas_url, temp_path/'gas.zip')
        # downloads from UCI, extracts and renames the dataset, and cleans up
        with zipfile.ZipFile(temp_path/'gas.zip') as zipped_file:
            zipped_file.extractall(temp_path/'gas-unzipped')
        with zipfile.ZipFile(temp_path/'gas-unzipped'/'HT_Sensor_dataset.zip') as zipped_file:
            zipped_file.extractall(temp_path)
        Path.rename(temp_path/'HT_Sensor_dataset.dat', final_filename)
        #clean-up
        shutil.rmtree(temp_path)
        print(f'Dataset saved at {final_filename}')

# Energy
def fetch_energy_data():
    final_filename = data_path/'energydata.csv'
    if not Path.exists(final_filename):
        print('Donwloading Energy Dataset')
        uci_energy_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv'
        response = urllib.request.urlretrieve(uci_energy_url, final_filename)
        print(f'Dataset saved at {final_filename}')

# COVID
def fetch_covid_data():
    print('Due to licencing issues, we are unable to share the COVID dataset. ' + 
          'Please contact the author via his email on the GitHub page for access to the data!')

    
data_path = Path('..') / '..' / 'real-world-datasets'
if not Path.exists(data_path):
    Path.mkdir(data_path)
    
if len(sys.argv) > 1:
    switch = sys.argv[1]
    if switch == 'all':
        fetch_gas_data()
        fetch_energy_data()
        fetch_covid_data()
    elif switch == 'gas':
        fetch_gas_data()
    elif switch == 'energy':
        fetch_energy_data()
    elif switch == 'covid':
        fetch_covid_data()
    else:
        print('Argument not recognized, please pass either: all, gas, energy, or covid.')
else:
    print('This script requires an argument to run. To fetch the data please pass either: all, gas, energy, or covid.')
