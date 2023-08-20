import os
from typing import List, Dict, Optional, Any

import asyncio
import aiohttp
import polars as pl
import numpy as np


class NYTimes:
    def __init__(
        self, 
        api_key: str = os.environ["NYTIMES_TECH_API_KEY"],
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


    async def _fetch_machine_learning_articles(
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
            async with session.get(BASE_URL, params=query) as response:
                if response.status != 200:
                    raise Exception(f"GET {BASE_URL} {response.status}") 
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
            await self._fetch_machine_learning_articles()
        )
        total_hits: int = first_page["response"]["meta"]["hits"]
        total_pages: int = min(
            (total_hits // 10) + (1 if total_hits % 10 else 0), 
            100,
        )

        # Create tasks to fetch all pages concurrently
        tasks: List[Any] = [
            self._fetch_machine_learning_articles(page) 
            for page in range(1, total_pages)
        ]
        pages: List[Dict[str, Any]] = await asyncio.gather(*tasks)

        # Aggregate all the articles
        all_articles: List[Dict[str, Any]] = first_page["response"]["docs"]
        for page in pages:
            all_articles.extend(page["response"]["docs"])

        return all_articles


    async def impute_article_info_into_dataframe(
        self,
        articles: List[Dict[str, Any]],
    ) -> pl.DataFrame:
        """
        Convert NYTimes articles data into a Polars DataFrame.

        Parameters
        ----------
        articles: List[Dict[str, Any]]
            A list of dictionaries, each representing an article from the NY Times API.

        Returns
        -------
        pl.DataFrame
            A Polars DataFrame representation of the articles data.
        """
        
        # Extracting column data
        main_headlines: List[Optional[str]] = [
            art["headline"]["main"] for art in articles
        ]
        abstracts: List[Optional[str]] = [
            art["abstract"] for art in articles
        ]
        web_urls: List[Optional[str]] = [
            art["web_url"] for art in articles
        ]
        snippets: List[Optional[str]] = [
            art["snippet"] for art in articles
        ]
        lead_paragraphs: List[Optional[str]] = [
            art["lead_paragraph"] for art in articles
        ]
        pub_dates: List[Optional[str]] = [ 
            art["pub_date"] for art in articles
        ]
        document_types: List[Optional[str]] = [ 
            art["document_type"] for art in articles
        ]
        news_desks: List[Optional[str]] = [
            art["news_desk"] for art in articles
        ]
        section_names: List[Optional[str]] = [ 
            art["section_name"] for art in articles
        ]
        subsection_names: List[Optional[str]] = [
            art.get("subsection_name", None) for art in articles
        ]
        author_lists: List[Optional[List[str]]] = [
            " ".join([
                person['firstname'] + ' ' + person['lastname'] 
                for person in art["byline"]["person"]
            ]) or np.nan 
            for art in articles
        ]
        organization_lists: List[Optional[List[str]]] = [
            " ".join(art["byline"]["organization"]) or np.nan 
            if art["byline"]["organization"] else np.nan 
            for art in articles
        ]
        type_of_materials: List[Optional[str]] = [
            art["type_of_material"] for art in articles
        ]
        word_counts: List[Optional[str]] = [
            art["word_count"] for art in articles
        ]

        # Creating a DataFrame with indicated fields
        return pl.DataFrame({
            "main_headline": main_headlines,
            "abstract": abstracts,
            "web_url": web_urls,
            "snippet": snippets,
            "lead_paragraph": lead_paragraphs,
            "pub_date": pub_dates,
            "document_type": document_types,
            "news_desk": news_desks,
            "section_name": section_names,
            "subsection_name": subsection_names,
            "author_list": author_lists,
            "organization_list": organization_lists,
            "type_of_material": type_of_materials,
            "word_count": word_counts,
        })
