from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import pandas as pd
from io import StringIO
import time

chrome_options = Options()
chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
driver = webdriver.Chrome(options=chrome_options)
driver.execute_script("window.open('');")
driver.switch_to.window(driver.window_handles[-1])

years = [2012, 2013, 2014, 2015, 2016, 2017, 2018,
         2019, 2020, 2021, 2022, 2023, 2024]

for year in years:
    url = "https://stathead.com/football/player-game-finder.cgi?request=1&order_by=fantasy_points_ppr&\
           timeframe=seasons&year_min=" + str(year) + "&year_max=" + str(year)
    driver.get(url)
    time.sleep(3)

    csv_button = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Get table as CSV')]"))
    )

    driver.execute_script("arguments[0].click();", csv_button)
    time.sleep(0.5)

    csv_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "csv_stats"))
    )

    csv_text = csv_element.text

    lines = csv_text.strip().split('\n')
    csv_data = '\n'.join(lines[5:])

    df = pd.read_csv(StringIO(csv_data))

    df.drop(columns=["FantPt", "PPR.1", "DKPt", "FDPt"], inplace=True)
    df.rename(columns={'-9999': 'PFR_Id'}, inplace=True)
    df.rename(columns={'Unnamed: 9': 'Home_Away'}, inplace=True)

    # Fill and replace Home_Away values
    df['Home_Away'] = df['Home_Away'].fillna("Home")
    df['Home_Away'] = df['Home_Away'].replace("@", "Away")

    while df['PPR'].min() > 0.00:
        next_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//a[contains(text(), 'Next Page')]"))
        )

        new_url = next_button.get_attribute('href')

        driver.get(new_url)
        time.sleep(3)

        csv_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Get table as CSV')]"))
        )

        driver.execute_script("arguments[0].click();", csv_button)
        time.sleep(0.5)

        csv_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "csv_stats"))
        )

        csv_text = csv_element.text

        lines = csv_text.strip().split('\n')
        csv_data = '\n'.join(lines[5:])

        temp_df = pd.read_csv(StringIO(csv_data))

        temp_df.drop(columns=["FantPt", "PPR.1", "DKPt", "FDPt", "Rk"], inplace=True)
        temp_df.rename(columns={'-9999': 'PFR_Id'}, inplace=True)
        temp_df.rename(columns={'Unnamed: 9': 'Home_Away'}, inplace=True)

        # Fill and replace Home_Away values
        temp_df['Home_Away'] = temp_df['Home_Away'].fillna("Home")
        temp_df['Home_Away'] = temp_df['Home_Away'].replace("@", "Away")

        df = pd.concat([df, temp_df])

    df = df[df['PPR'] > 0.00]

    # Optional: Save to CSV
    output_file = "weekly_fantasy_points/" + str(year) + ".csv"
    df.to_csv(output_file, index=False)