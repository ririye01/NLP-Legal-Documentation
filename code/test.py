from nytimes.client_nytimes import NYTimes
import os

key = os.environ["NYTIMES_TECH_API_KEY"]
secret = os.environ["NYTIMES_TECH_API_SECRET"]

ny = NYTimes(key, secret)

async def main():
    ny.get_articles_from_specified_month(7, 2023)
