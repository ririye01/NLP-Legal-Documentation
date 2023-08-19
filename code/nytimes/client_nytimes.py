import asyncio
from typing import List, Dict, Optional, Any
import aiohttp
import os


class NYTimes:
    def __init__(self, 
        api_key: str = os.environ["NYTIMES_TECH_API_KEY"]
    ) -> None:
      self._api_key = api_key

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
        url: str = f"https://api.nytimes.com/svc/archive/v1/{year}/" + \
                   f"{month}.json?api-key={self._api_key}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"GET {url} {response.status}")
                return await response.json()


    async def fetch_machine_learning_articles(
        self, page: int = 0
    ) -> Dict[str, Any]:
        """
        Fetch a page of articles based on predefined query terms and date filter.
        
        Parameters
        ----------
        page : int, optional
            Page number to fetch, by default 0.
        
        Returns
        -------
        Dict[str, Any]
            A dictionary representing the JSON response from the NY Times API.
        """
        # Identify machine learning query terms 
        BASE_URL: str = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
        QUERY_TERMS: List[str] = [
            "machine learning", 
            "artificial intelligence", 
            "ai", 
            "chatgpt",
        ]
        DATE_FILTER: str = "pub_date:(2000-01-01T00:00:00Z TO *)"

        # Formulate the query string for the terms
        query_string: str = " OR ".join(
            ['"' + term + '"' for term in QUERY_TERMS]
        )
        query: Dict[str, str] = {
            "q": query_string,
            "fq": DATE_FILTER,
            "api-key": self._api_key,
            "page": str(page),
        }

        async with aiohttp.ClientSession() as session:
            response = session.get(BASE_URL, params=query)
            return await response.json()


    async def gather_all_machine_learning_articles(self) -> List[Dict[str, Any]]:
        """
        Aggregate all articles across pages.
        
        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries, each representing an article from the NY Times API.
        """
        # Start by fetching the first page to get the total hits
        first_page: Dict[str, Any] = (
            await self.fetch_machine_learning_articles()
        )
        total_hits: int = first_page["response"]["meta"]["hits"]
        total_pages: int = min(
            (total_hits // 10) + (1 if total_hits % 10 else 0), 
            100,
        )

        # Create tasks to fetch all pages concurrently
        tasks: List[Any] = [
            self.fetch_machine_learning_articles(page) 
            for page in range(1, total_pages)
        ]
        pages: List[Dict[str, Any]] = await asyncio.gather(*tasks)

        # Aggregate all the articles
        all_articles: List[Dict[str, Any]] = first_page["response"]["docs"]
        for page in pages:
            all_articles.extend(page["response"]["docs"])

        return all_articles
