from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

def scroll_container(driver, container, pause=0.5):
    last_height = driver.execute_script("return arguments[0].scrollHeight", container)
    while True:
        driver.execute_script("arguments[0].scrollTo(0, arguments[0].scrollHeight);", container)
        time.sleep(pause)
        new_height = driver.execute_script("return arguments[0].scrollHeight", container)
        if new_height == last_height:
            break
        last_height = new_height

def scrape_kyber_table(driver, url):
    driver.get(url)
    time.sleep(20)  # give React a few seconds to render everything

    page_size_elems = driver.find_elements(By.CSS_SELECTOR,
                                        ".kyber-filter-strip__option")

    largest_page_size = 0
    largest_page_size_element = None
    for page_size in page_size_elems:
        if int(page_size.text.strip()) > largest_page_size:
            largest_page_size_element = page_size
            largest_page_size = int(page_size.text.strip())

    largest_page_size_element.click()

    # grab headers directly
    header_elems = driver.find_elements(By.CSS_SELECTOR,
                                        ".kyber-table-header__column span.kyber-mod--overflow-ellipsis")
    headers = [h.text.strip() for h in header_elems if h.text.strip()]

    print(headers)

    # QBs
    #headers = ['RANK', 'PLAYER', '#', 'POS', '#G', 'TEAM', 'DB', 'ATT', 'COM', 'COM%', 'YDS', 'YPA', 'TD', 'INT', 'OFF',
    #           'PASS', 'RUN', 'FUM', 'BTT', 'BTT%', 'TWP', 'TWP%', 'ADOT', 'ADJ%', 'DRP', 'DRP%', 'BAT', 'HAT', 'TA',
    #           'DPR', 'SK', 'P2S%', 'TTT', 'SCR', '1ST', 'NFL']

    # WR &B TEs
    #headers = ['RANK', 'PLAYER', '#', 'POS', '#G', 'TEAM', 'TGT', 'REC', 'REC%', 'YDS', 'Y/REC', 'TD', 'OFF', 'RECV',
    #           'DROP', 'FUM', 'PBLK', 'PASS', 'RECV', 'RT%', 'PBLK', 'PB%', 'SLOT', 'SLT%', 'WIDE', 'WID%', 'INL',
    #           'INL%', 'YAC', 'YAC/REC', 'Y/RR', 'ADOT', 'LNG', 'DRP', 'DRP%', 'CTT', 'CTC', 'CTC%', 'INT', 'FUM',
    #           'MTF', '1ST', 'RTG']

    headers = ['RANK', 'PLAYER', '#', 'POS', '#G', 'TEAM', 'SNP', 'ATT', 'YDS', 'YPA', 'TD', 'FUM', 'OFF', 'RUN', 'FUM',
               'RBLK', 'YCO', 'YCO/A', 'MTF', 'LNG', '10+', 'ZONE', 'GAP', 'SCR', 'SYDS', 'DYDS', 'D15+', 'BAY', 'BAY%',
               '1ST', 'PEN', 'RECV', 'PBLK', 'TGT', 'REC', 'YDS', 'RSNP', 'Y/RR', 'DRP', 'ELU']

    def extract_data(passed_data):
        # grab rows
        row_elems = driver.find_elements(By.CSS_SELECTOR, ".kyber-table-body__row")
        n = len(row_elems) // 2  # assume exactly half sticky, half main

        sticky_rows = row_elems[:n]
        main_rows = row_elems[n:]


        if passed_data:
            my_data = passed_data
        else:
            my_data = []
        for sticky, main in zip(sticky_rows, main_rows):
            # sticky left column
            sticky_values = [c.text.strip() for c in sticky.find_elements(By.CSS_SELECTOR, ".kyber-table-body-cell")]

            # main columns
            main_values = [c.text.strip() for c in main.find_elements(By.CSS_SELECTOR, ".kyber-table-body-cell")]

            row_data = (sticky_values + main_values)[:-1]
            my_data.append(row_data)
        return my_data

    data = extract_data(None)

    #next_page_element = driver.find_elements(By.CSS_SELECTOR, ".kyber-table-pagination__button-next")

    next_page_element = []
    print(next_page_element)

    if len(next_page_element) > 0:
        next_page_element[0].click()
        time.sleep(20)
        data = extract_data(data)

    if data:
        df = pd.DataFrame(data, columns=headers[:len(data[0])])
    else:
        df = pd.DataFrame(columns=headers)
    return df

def main():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)

    try:
        driver.get("https://auth.pff.com")
        print("Log in manually, then press Enter here...")
        input()

        years = ["2011", "2012", "2013", "2014", "2015",
                 "2016", "2017", "2018", "2019", "2020",
                 "2021", "2022", "2023", "2024"]
        # years = ["2015"]
        for year in years:
            print(year)
            url = "https://premium.pff.com/nfl/positions/" + year + "/REGPO/rushing?position=HB,FB"
            df = scrape_kyber_table(driver, url)

            df.to_csv("pff_" + year + "_rbs.csv", index=False)

    finally:
        driver.quit()

if __name__ == "__main__":
    main()
