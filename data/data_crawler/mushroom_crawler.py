import os
import pandas as pd
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup
import time
from PIL import Image

# Don't cause too much load on the server
REQUEST_RATE_LIMITER = 0.5

@dataclass
class CsvFileEntry:
  edibility: str
  mushroom: str # species
  csv_path: str

  def __repr__(self):
    return f"{self.edibility}, {self.mushroom}, {self.csv_path}"
  
  def __str__(self):
    return self.__repr__()


class MushroomCrawler:
  default_base_path = os.path.join(os.getcwd(), "data", "data_crawler", "csv_files")
  default_edibilities = ["deadly", "edible", "not_edible", "poisonous"]
  output_dir_path = os.path.join(os.getcwd(), "data", "data_crawler", "output")

  def __init__(self, base_path: str = default_base_path, edibilities: list[str] = default_edibilities, output_dir_path: str = output_dir_path):
    self.base_path = base_path
    self.edibilities = edibilities
    self.output_dir_path = output_dir_path
    self.csv_file_entries = self.init_csv_file_entries()
    
  def init_csv_file_entries(self) -> list[CsvFileEntry]:
    csv_file_entries: list[CsvFileEntry] = []
    for edibility in self.edibilities:
      edibility_path = os.path.join(self.base_path, edibility)
      csv_file_entry = self.get_folder_csv_file_entries(edibility, edibility_path)
      csv_file_entries.extend(csv_file_entry)

    if len(self.edibilities) == 0:
      edibility = "undefined"
      edibility_path = self.base_path
      csv_file_entry = self.get_folder_csv_file_entries(edibility, edibility_path)
      csv_file_entries.append(csv_file_entry)


    return csv_file_entries
  
  def get_folder_csv_file_entries(self, edibility: str, edibility_path: str) -> list[CsvFileEntry]:
    csv_files = [file for file in os.listdir(edibility_path) if file.endswith('.csv')]
    csv_file_entries: list[CsvFileEntry] = [] 
    for csv_file in csv_files:
      mushroom = csv_file.replace(".csv", "")
      csv_file_entry = CsvFileEntry(edibility, mushroom, os.path.join(edibility_path, csv_file))
      csv_file_entries.append(csv_file_entry)
    return csv_file_entries
  
  def crawl(self, mushroom: str = None):
    """
    Crawls through all CSV files and processes them. On average, CSV files have 100s of entries and each entry causes
    a request to the server to retrieve each "observation". Images of the observation need to be requested and saved.

    *Note* that using requests only retrieves a single image. I guess we would need to use Selenium to fix that.

    :param str mushroom: If provided, only crawl entries for this specific mushroom species. Defaults to None which crawls all mushrooms.
    """
    for csv_file_entry in self.csv_file_entries:
      if mushroom and csv_file_entry.mushroom != mushroom:
        continue
      self.crawl_csv_file_entry(csv_file_entry)

  def crawl_csv_file_entry(self, csv_file_entry: CsvFileEntry):
    df = pd.read_csv(csv_file_entry.csv_path)
    
    if "crawled" not in df.columns:
      df["crawled"] = 0

    for index, row in df.iterrows():
      if (index+1) % 100 == 0 or index == len(df)-1:
        print(f"Crawling row {index+1} of {len(df)} [{csv_file_entry.mushroom} ({csv_file_entry.edibility})]")
      if df.at[index, "crawled"] != 0:
        continue
      
      self.process_row(row, csv_file_entry)

      if index == 6:
        df.to_csv(csv_file_entry.csv_path, index=False)
        break
    
    # df.to_csv(csv_file_entry.csv_path, index=False)
  
  def process_row(self, row: pd.Series, csv_file_entry: CsvFileEntry):
    allowed_validation_statuses = ["approved", "expert approved"]
    validation_status = row["validationStatus"]

    if validation_status not in allowed_validation_statuses:
      row["crawled"] = 2
      return
    row["crawled"] = 1

    row_id = row["_id"]
    row_uri = row["URI"]
    html_content = self.get_uri_content(row_uri)
    image_urls = self.get_mushroom_image_urls(html_content)

    # Create directory structure if it doesn't exist
    image_dir = os.path.join(self.output_dir_path, csv_file_entry.edibility, csv_file_entry.mushroom)
    os.makedirs(image_dir, exist_ok=True)

    for index, image_url in enumerate(image_urls):
      try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        
        img = Image.open(response.raw)
        filename = row_id if index == 0 else f"{row_id}_{index+1}"
        img.save(os.path.join(image_dir, f"{filename}.jpg"))
      
        time.sleep(REQUEST_RATE_LIMITER)  # Respect rate limiting
        
      except requests.RequestException as err:
        print(f"Error downloading image {image_url}: {str(err)}")
        continue
    print(image_urls)

  def get_uri_content(self, uri: str):
    try:
      res = requests.get(uri)
      time.sleep(REQUEST_RATE_LIMITER)
      res.raise_for_status()
      content = res.text
      return content
    except requests.RequestException as err:
      print(f"Error fetching URI {uri}: {str(err)}")
      return None
  
  def get_mushroom_image_urls(self, html_content: str):
    if not html_content:
      return []
        
    soup = BeautifulSoup(html_content, "html.parser")
    image_urls = []
    
    # Find all og:image meta tags
    og_images = soup.find_all("meta", property="og:image")
    for og_image in og_images:
      url = og_image.get("content")
      if url:
        image_urls.append(url)
            
    return list(set(image_urls))  # Remove duplicates


if __name__ == "__main__":
  crawler = MushroomCrawler()
  # crawler.crawl()

  chanterelle_entry = [entry for entry in crawler.csv_file_entries if entry.mushroom == "Craterellus_cinereus"][0]
  crawler.crawl_csv_file_entry(chanterelle_entry)
