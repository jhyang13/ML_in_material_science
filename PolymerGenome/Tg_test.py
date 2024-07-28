# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 12:16:22 2021

@author: let20002
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import ElementClickInterceptedException
import time
import sys
import re
import requests
from bs4 import BeautifulSoup
from math import ceil
import pandas as pd


import pickle
import pandas as pd
from rdkit import Chem
import numpy as np

def sleepTime():
    return random.uniform(4.5, 8.1)

driver = webdriver.Chrome(executable_path=r"C:\Users\let20002\Downloads\chromedriver.exe")
driver.get("https://www.polymergenome.org/predict/index.php?m=11")

username =  'weikang.xian@uconn.edu'
password = 'YingLiLab123456'

username_textbox = driver.find_element_by_id("PyPG_account_username")
username_textbox.send_keys(username)

password_textbox = driver.find_element_by_id("PyPG_account_passcode")
password_textbox.send_keys(password)

login_button = driver.find_elements_by_xpath('/html/body/table/tbody/tr[2]/td/center/div/table/tbody/tr/td/table/tbody/tr[5]/td/center/input')[0]
login_button.click()

#dataset_2 = pickle.load(open("dataset_2_PolymerGenome.pkl","rb"))
df = pd.read_csv("600_MolFile.txt", sep='\t', header=None)

for i in range(3050,len(dataset_2.Smiles)):
    smile = dataset_2.Smiles.iloc[i]
    try:
        time.sleep(1)
        driver.get("https://www.polymergenome.org/predict/index.php?m=11")
        
        smile_bracket = smile.replace('*','[*]')
        pageNumber = driver.find_elements_by_xpath('/html/body/table/tbody/tr[3]/td/center/div/center/table/tbody/tr/td/center/table/tbody/tr/td[2]/p/input')[0]
        pageNumber.clear()
        pageNumber.send_keys(smile_bracket)

        page_button = driver.find_elements_by_xpath('/html/body/table/tbody/tr[3]/td/center/div/center/table/tbody/tr/td/center/table/tbody/tr/td[3]/input')[0]
        page_button.click()
        
        Tg_html = driver.find_elements_by_xpath('/html/body/table/tbody/tr[3]/td/center/center/table/tbody/tr[2]/td/center/table[1]/tbody/tr[1]/td[3]/table/tbody/tr[4]/td[3]/iframe')[0]
        src = Tg_html.get_attribute("src")
        driver.get(src)
        Tg_value = driver.find_elements_by_xpath('/html/body/table/tbody/tr/td/font')[0]
        dataset_2['Tg_pred'].iloc[i] = Tg_value.text.split(' K')[0].split(' ± ')[0]
        dataset_2['Tg_uncertainty'].iloc[i] = Tg_value.text.split(' K')[0].split(' ± ')[1]
        print(i, dataset_2['Tg_pred'].iloc[i])
    except:
        print('query failed for smile: {}'.format(smile))
        dataset_2['Tg_pred'].iloc[i] = np.nan
        dataset_2['Tg_uncertainty'].iloc[i] = np.nan