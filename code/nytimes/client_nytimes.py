from typing import List, Dict
import aiohttp


class NYTimes:
    def __init__(
        self, api_key: str, api_secret: str,
    ) -> None:
      self._api_key = api_key
      self._api_secret = api_secret

    async def get_articles_from_specified_month(
        self, month: int, year: int,
    ) -> List[Dict]:
        """
        Retrieve article data from NYTimes API for all articles published
        in a specified month and year.

        Parameters
        ----------
        month: int
            Indicated month where user wants articles to be scraped from 
        year: int
            Indicated year where user wants articles to be scraped from 
            
        Returns
        -------
        List[Dict]
            Requested data in specified month-year timeframe
        """
        url: str = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={self._api_key}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"GET {url} {response.status}")
                return await response.json()
