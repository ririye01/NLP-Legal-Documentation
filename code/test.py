from nytimes.client_nytimes import NYTimes
import os
import asyncio

ny = NYTimes()

async def main():
    ny.gather_all_machine_learning_articles()

if __name__ == "__main__":
    asyncio.run(main())
