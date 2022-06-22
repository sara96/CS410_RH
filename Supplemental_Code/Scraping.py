import os
import time

import requests
import io
import hashlib

from PIL import Image
from rsa import sign
from selenium import webdriver
from selenium.webdriver.common.by import By
from sys import platform
import signal
from glob import glob


# source: https://ladvien.com/scraping-internet-for-magic-symbols/
number_of_images = 3000
GET_IMAGE_TIMEOUT = 2
SLEEP_BETWEEN_INTERACTIONS = 0.1
SLEEP_BEFORE_MORE = 5
IMAGE_QUALITY = 85

output_path = "Scraped_Images"


# check if image is in there
dirs = glob(output_path + "*")
dirs = [dir.split("/")[-1].replace("_", " ") for dir in dirs]

Keywords = list(map(str, input(
    "What images would you like to search and scrape?\n").strip().split()))


Keywords = [term for term in Keywords if term not in dirs]

wd = webdriver.Chrome()


# fetching images
def fetch_image_urls(
    query: str,
    max_links: int,
    wd: webdriver,
    sleep_between_interactions: int = 1,
):

    # scrolling
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)

    # google query
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # loading the page
    wd.get(search_url.format(q=query))

    # Declare as set to prevent dup.
    image_urls = set()
    image_count = 0  # count the amount of image from each key till the max set
    results_start = 0

    while image_count < max_links:
        scroll_to_end(wd)

        # get image
        image_res = wd.find_elements(By.CSS_SELECTOR, value="img.Q4LuWd")

        number_results = len(image_res)
        print(
            f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")

        # loop throught the image
        for img in image_res[results_start:number_results]:
            # click on thumbnail to get image from it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract img urls
            actual_images = wd.find_elements(
                By.CSS_SELECTOR, value="img.n3VNCb")
            for actual_image in actual_images:
                if actual_image.get_attribute(
                    "src"
                ) and "http" in actual_image.get_attribute("src"):
                    image_urls.add(actual_image.get_attribute("src"))

            image_count = len(image_urls)

            # if reach the max res need to get terminate
            if len(image_urls) >= max_links:
                print(f"Found: {len(image_urls)} image links, done!")
                break
        else:
            # else keep looking for more
            print("Found:", len(image_urls),
                  "image links, looking for more ...")
            time.sleep(SLEEP_BEFORE_MORE)

            not_what_you_want_button = ""
            try:
                not_what_you_want_button = wd.find_element_by_css_selector(
                    ".r0zKGf")
            except:
                pass

                # if no more images
            if not_what_you_want_button:
                print("No more images available.")
                return image_urls

                # look to it
            load_more_button = wd.find_element_by_css_selector(".mye4qd")
            if load_more_button and not not_what_you_want_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")

        results_start = len(image_res)
    return image_urls


# downloading the images
def persist_image(folder_path: str, url: str):
    try:
        print("Getting image")
        # Download the image.  If timeout is exceeded, throw an error.
        # with timeout(GET_IMAGE_TIMEOUT):
        image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        # Convert the image into a bit stream, then save it.
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert("RGB")
        # Create a unique filepath from the contents of the image.
        file_path = os.path.join(
            folder_path, hashlib.sha1(image_content).hexdigest()[:10] + ".jpg"
        )
        with open(file_path, "wb") as f:
            image.save(f, "JPEG", quality=IMAGE_QUALITY)
        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")


def search_and_download(search_term: str, target_path="\Scraped_Images", number_images=5):
    # Create a folder name.
    target_folder = os.path.join(
        target_path, "_".join(search_term.lower().split(" ")))

    # Create image folder if needed.
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Open Chrome
    with webdriver.Chrome() as wd:
        # Search for images URLs.
        res = fetch_image_urls(
            search_term,
            number_images,
            wd=wd,
            sleep_between_interactions=SLEEP_BETWEEN_INTERACTIONS,
        )

        # Download the images.
        if res is not None:
            for elem in res:
                persist_image(target_folder, elem)
        else:
            print(f"Failed to return links for term: {search_term}")


for term in Keywords:
    search_and_download(term, output_path, number_of_images)
