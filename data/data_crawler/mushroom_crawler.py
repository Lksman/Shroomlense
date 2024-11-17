"""
This module creates the dataset. Steps:
0) Initialize the MushroomCrawler object.
1) Read all CSV files of the specified folder structure.
2) Use Selenium with the URI of each CSV row to download images.
3) Resize the shortest side of each image to 256px.
4) Save compressed images to the specified output directory.

Notes:
- The CSV files keep track of which observations have already been handled/downloaded.
- Selenium is used because doing a simple request only retrieves simplified HTML with only a single image per "observation".
- Random sleeps are added to reduce load on the server and make crawling "less suspicious".
  - This also means that crawling takes much longer but we got enough time.
- Some classes have much more data than others, like 150 vs 3000 images
"""

import os
import time
import random
from dataclasses import dataclass
from PIL import Image

import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


CRAWL_DELAY_SEC = (1, 2)
CRAWL_TIMEOUT_SEC = 30

CRAWL_STATUS_NOT_CRAWLED = 0
CRAWL_STATUS_SUCCESS = 1
CRAWL_STATUS_BAD_OBSERVATION = 2
CRAWL_STATUS_FAILED = -1

@dataclass
class MushroomCatalog:
  edibility: str
  mushroom_name: str
  csv_path: str

  def __repr__(self):
    return f"{self.edibility}, {self.mushroom_name}, {self.csv_path}"
  
  def __str__(self):
    return self.__repr__()


class MushroomCrawler:
  default_base_path = os.path.join(os.getcwd(), "data", "data_crawler", "csv_files")
  default_edibilities = ["deadly", "edible", "not_edible", "poisonous"]
  output_dir_root = os.path.join(os.getcwd(), "data", "data_crawler", "output")

  def __init__(self, base_path: str = default_base_path, edibilities: list[str] = default_edibilities, output_dir_path: str = output_dir_root):
    self.base_path = base_path
    self.edibilities = edibilities
    self.output_dir_root = output_dir_path
    self.mushroom_catalogs = self.init_mushroom_catalogs()
    self.web_driver = self.init_web_driver()
    self.rerun_failed = False # If True, tries to rerun failed observations again.

  def init_mushroom_catalogs(self) -> list[MushroomCatalog]:
    mushroom_catalogs: list[MushroomCatalog] = []
    for edibility in self.edibilities:
      edibility_path = os.path.join(self.base_path, edibility)
      mushroom_catalog = self.create_mushroom_catalogs(edibility, edibility_path)
      mushroom_catalogs.extend(mushroom_catalog)

    if len(self.edibilities) == 0:
      edibility = "undefined"
      edibility_path = self.base_path
      mushroom_catalog = self.create_mushroom_catalogs(edibility, edibility_path)
      mushroom_catalogs.append(mushroom_catalog)

    return mushroom_catalogs
  
  def create_mushroom_catalogs(self, edibility: str, edibility_path: str) -> list[MushroomCatalog]:
    """
    Creates a list of MushroomCatalog objects for each CSV file in the given edibility folder.
    """
    csv_files = [file for file in os.listdir(edibility_path) if file.endswith('.csv')]
    mushroom_catalogs: list[MushroomCatalog] = [] 
    for csv_file in csv_files:
      mushroom = csv_file.replace(".csv", "")
      mushroom_catalog = MushroomCatalog(edibility, mushroom, os.path.join(edibility_path, csv_file))
      mushroom_catalogs.append(mushroom_catalog)
    return mushroom_catalogs

  def set_rerun_failed(self, rerun_failed: bool):
    self.rerun_failed = rerun_failed
  
  def crawl(self, mushroom: str = None):
    """
    Crawls through all CSV files and processes them. On average, CSV files have 100s of entries and each entry causes
    a request to the server to retrieve each "observation". Images of the observation need to be requested and saved.

    *Note* that using requests only retrieves a single image. I guess we would need to use Selenium to fix that.

    :param str mushroom: If provided, only crawl entries for this specific mushroom species. Defaults to None which crawls all mushrooms.
    """
    for mushroom_catalog in self.mushroom_catalogs:
      if mushroom and mushroom_catalog.mushroom_name != mushroom:
        continue
      self.crawl_mushroom_catalog(mushroom_catalog)

  def init_web_driver(self):
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    web_driver = webdriver.Chrome(options=chrome_options)

    # timeout settings
    web_driver.set_page_load_timeout(CRAWL_TIMEOUT_SEC)
    web_driver.set_script_timeout(CRAWL_TIMEOUT_SEC)   

    return web_driver

  def crawl_mushroom_catalog(self, mushroom_catalog: MushroomCatalog):
    """
    Iterate through each row of the CSV file and process the URI.
    """

    df = pd.read_csv(mushroom_catalog.csv_path)
    
    if "crawled" not in df.columns:
      df["crawled"] = CRAWL_STATUS_NOT_CRAWLED

    print(f"Crawling {mushroom_catalog.mushroom_name} ({mushroom_catalog.edibility})")
    for index, row in df.iterrows():
      # update progress bar
      percentage = round(((index+1) / len(df)) * 100)
      print(f"\r{self.progress_percentage_to_string(percentage)} {index+1} of {len(df)} {row['_id']}", end='', flush=True)

      crawl_status = int(df.at[index, "crawled"])
      if crawl_status != CRAWL_STATUS_NOT_CRAWLED:
        if crawl_status == CRAWL_STATUS_FAILED: 
          if not self.rerun_failed:
            continue
        else:
          continue

      success_state = self.process_row(row, mushroom_catalog)
      df.at[index, "crawled"] = success_state

      # periodically save progress
      if index > 0 and index % 20 == 0:
        df.to_csv(mushroom_catalog.csv_path, index=False)
    
    df.to_csv(mushroom_catalog.csv_path, index=False)
    print(f"Finished crawling {mushroom_catalog.mushroom_name} ({mushroom_catalog.edibility})")
  
  def process_row(self, row: pd.Series, mushroom_catalog: MushroomCatalog):
    allowed_validation_statuses = ["approved", "expert approved"]
    validation_status = row["validationStatus"]

    if validation_status not in allowed_validation_statuses:
      return CRAWL_STATUS_BAD_OBSERVATION
    
    if not self.request_uri(row["URI"]):
      return CRAWL_STATUS_FAILED
    
    image_urls = self.get_mushroom_image_urls()
    output_directory = os.path.join(self.output_dir_root, mushroom_catalog.edibility, mushroom_catalog.mushroom_name)
    if not self.download_images(image_urls, row["_id"], output_directory):
      return CRAWL_STATUS_FAILED
    
    return CRAWL_STATUS_SUCCESS
  
  def sleep(self):
    """
    Sleep to prevent overloading the server. Define multiple sleep conditions to make crawling less suspicious.
    """
    seconds = random.uniform(CRAWL_DELAY_SEC[0], CRAWL_DELAY_SEC[1])
    if random.random() < 0.1:
      seconds += random.uniform(0, 2)
    if random.random() < 0.01:
      seconds += random.uniform(3, 10)
    if random.random() < 0.001:
      seconds += random.uniform(20, 40)

    time.sleep(seconds)

  def request_uri(self, uri: str):
    try:
      self.web_driver.get(uri)
      self.sleep()
      return True
    except Exception as err:
      print(f"Error fetching URI {uri}: {str(err)}")
      self.sleep()
      return False
  
  def get_mushroom_image_urls(self):
    """
    Images are contained within a grid tile element.
    """
    try:
      image_urls = []
      grid_tiles = self.web_driver.find_elements(By.CSS_SELECTOR, "md-grid-tile")
      
      for tile in grid_tiles:
        style = tile.get_attribute("style")
        if style and "background-image" in style:
          url_start = style.find('url("') + 5
          url_end = style.find('")', url_start)
          url = style[url_start:url_end]
          if url:
            image_urls.append(url)
      
      unique_image_urls = list(set(image_urls))
      return unique_image_urls
    except Exception as err:
      print(f"Error extracting image URLs: {str(err)}")
      return []
  
  def download_images(self, image_urls: list[str], row_id: str, output_directory: str):
    # Create directory structure if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    mushroom_name = os.path.basename(output_directory)
    success = True

    try:
      for index, image_url in enumerate(image_urls):
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        
        image = Image.open(response.raw)
        image = self.process_image(image)
        filename = row_id if index == 0 else f"{row_id}_{index+1}"
        filename = f"{mushroom_name}_{filename}.jpg"
        image.save(os.path.join(output_directory, filename), optimize=True, quality=75)
        self.sleep()
      return success
    except Exception as err:
      print(f"Error downloading image {image_url}:\n {str(err)}")
      return not success

  def process_image(self, image: Image.Image, size: int = 256):
    width, height = image.size

    if image.mode != "RGB":
      image = image.convert("RGB")
    
    # Resize based on the shorter side
    if width < height:
        new_width = size
        new_height = int((height / width) * size)
    else:
        new_width = int((width / height) * size)
        new_height = size
    
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

  def progress_percentage_to_string(self, percentage: int):
    """
    Printing the percentage in color is of utmost importance.
    """

    string = ""
    if percentage < 20:
      string += "\033[31m" # red
    elif percentage < 40:
      string += "\033[35m" # purple
    elif percentage < 60:
      string += "\033[34m" # blue
    elif percentage < 80:
      string += "\033[36m" # cyan
    else:
      string += "\033[32m" # green

    string += f"[{percentage}%]\033[0m"
    return string

if __name__ == "__main__":
  crawler = MushroomCrawler()
  crawler.set_rerun_failed(False) # uncomment to rerun failed observations
  crawler.crawl() # crawls all mushrooms
  # crawler.crawl("Craterellus_cinereus") # crawls a specific mushroom