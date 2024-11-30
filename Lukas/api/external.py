import wikipedia
import bs4
import requests
from flask_restx import Resource, Namespace
from src.config import get_logger


logger = get_logger(__name__)
external_ns = Namespace("external", description="External Web operations")

@external_ns.route("/wikipedia/<string:mushroom_name>/table")
class WikipediaTable(Resource):
    def get(self, mushroom_name: str):
        try:
            logger.info(f"Getting table for {mushroom_name}")
            page = wikipedia.page(mushroom_name) # only checks if page exists, we request the html content later
            response = requests.get(page.url).text
            soup = bs4.BeautifulSoup(response, "html.parser")
            table = soup.find("table", {"class": "infobox"})
            if table is None:
                return "No table found", 404

            infobox_data = {}
            if table:
                for row in table.find_all("tr"):
                    cells = row.find_all(["td", "th"])
                    if len(cells) > 1:
                        infobox_data[cells[0].text.strip()] = cells[1].text.strip()
                        
            return infobox_data, 200
        
        except wikipedia.exceptions.PageError:
            logger.error(f"Mushroom not found: {mushroom_name}")
            return "Mushroom not found", 404
            
        except wikipedia.exceptions.DisambiguationError as e:
            logger.error(f"Disambiguation error: {e}")
            return f"Disambiguation error: {e}", 400

@external_ns.route("/wikipedia/<string:mushroom_name>/summary")
class WikipediaSummary(Resource):
    def get(self, mushroom_name: str):
        try:
            logger.info(f"Getting summary for {mushroom_name}")
            return wikipedia.summary(mushroom_name), 200
        
        except wikipedia.exceptions.PageError:
            logger.error(f"Mushroom not found: {mushroom_name}")
            return "Mushroom not found", 404
        
        except wikipedia.exceptions.DisambiguationError as e:
            logger.error(f"Disambiguation error: {e}")
            return f"Disambiguation error: {e}", 400
        
