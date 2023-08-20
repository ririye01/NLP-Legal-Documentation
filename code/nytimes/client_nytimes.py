import os
from typing import List, Dict, Optional, Union, Any

import asyncio
import aiohttp
import asyncpg
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
        existing_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """
        Convert NYTimes articles data into a Polars DataFrame.

        Parameters
        ----------
        articles: List[Dict[str, Any]]
            A list of dictionaries, each representing an article 
            from the NY Times API.

        existing_df: pl.DataFrame, optional
            An existing Polars DataFrame to append data to. If 
            not provided, a new DataFrame will be created.

        Returns
        -------
        pl.DataFrame
            A Polars DataFrame representation of the articles data, 
            either newly created or appended to the existing one.
        """
        
        # Extract data from articles
        data: Dict[str, List[Union[str, int, Optional[Any]]]] = {
            "main_headline": [],
            "abstract": [],
            "web_url": [],
            "snippet": [],
            "lead_paragraph": [],
            "pub_date": [],
            "document_type": [],
            "news_desk": [],
            "section_name": [],
            "subsection_name": [],
            "author_list": [],
            "organization_list": [],
            "type_of_material": [],
            "word_count": [],
        }
        
        for art in articles:
            data["main_headline"].append(art["headline"]["main"])
            data["abstract"].append(art["abstract"])
            data["web_url"].append(art["web_url"])
            data["snippet"].append(art["snippet"])
            data["lead_paragraph"].append(art["lead_paragraph"])
            data["pub_date"].append(art["pub_date"])
            data["document_type"].append(art["document_type"])
            data["news_desk"].append(art["news_desk"])
            data["section_name"].append(art["section_name"])
            data["subsection_name"].append(art.get("subsection_name", None))
            data["author_list"].append(
                " ".join([
                    person["firstname"] + ' ' + person["lastname"] 
                    for person in art["byline"]["person"]
                ]) or None
            )
            data["organization_list"].append(
                " ".join(art["byline"]["organization"]) or None 
                if art["byline"]["organization"] else None
            )
            data["type_of_material"].append(art["type_of_material"])
            data["word_count"].append(art["word_count"])
        
        # Create or append to DataFrame
        new_df: pl.DataFrame = pl.DataFrame(data)
        
        if existing_df is not None:
            return existing_df.vstack(new_df)
        else:
            return new_df


    async def impute_article_info_into_database(
        self,
        articles: List[Dict[str, Any]],
        db_config: Dict[str, str]
    ) -> None:
        """
        Store NYTimes articles data into a PostgreSQL Database.

        Parameters
        ----------
        articles: List[Dict[str, Any]]
            A list of dictionaries, each representing an article from 
            the NY Times API.
        
        db_config: Dict[str, str]
            Database configuration parameters, including "user", 
            "password", "database", "host", and "port".
        
        """
        
        # Create the connection
        conn = await asyncpg.connect(**db_config)
        
        # SQL command to insert data
        insert_sql = """
            INSERT INTO nytimes_articles (
                main_headline, abstract, web_url, snippet, 
                lead_paragraph, pub_date, document_type, news_desk,
                section_name, subsection_name, author_list, 
                organization_list, type_of_material, word_count
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14);
        """
        
        # Insert each article into the database
        for art in articles:
            await conn.execute(
                insert_sql,
                art["headline"]["main"],
                art["abstract"],
                art["web_url"],
                art["snippet"],
                art["lead_paragraph"],
                art["pub_date"],
                art["document_type"],
                art["news_desk"],
                art["section_name"],
                art.get("subsection_name", None),
                " ".join([
                    person['firstname'] + ' ' + person['lastname']
                    for person in art["byline"]["person"]
                ]) or None,
                " ".join(art["byline"]["organization"]) or None
                if art["byline"]["organization"] else None,
                art["type_of_material"],
                art["word_count"]
            )
        
        # Close the connection
        await conn.close() 
