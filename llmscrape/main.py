import asyncio
from typing import List
from crawl4ai import AsyncWebCrawler
from pydantic import BaseModel
from llama_index.core import Document
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate, ChatPromptTemplate, SummaryIndex
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from scrapper import StructuredAccumulate

class Article(BaseModel):
    title: str
    link: str

class Articles(BaseModel):
    articles: List[Article]

def accumulator(acc: Articles | None, data: Articles) -> Articles:
    if acc is None:
        return data
    return Articles(articles=acc.articles + data.articles)

async def extract_articles():
    url = 'https://news.ycombinator.com/'

    async with AsyncWebCrawler(verbose=True) as crawler:
        llm = Ollama(
            model="llama3.2",
            request_timeout=240,
            base_url="http://localhost:8080",
            temperature=0.0,
        )

        result = await crawler.arun(
            url=url,
            bypass_cache=True,
        )

        index = SummaryIndex.from_documents(
            documents=[Document(text=result.markdown)],
        )

        engine = index.as_query_engine(
            llm=llm,
            response_synthesizer=StructuredAccumulate(llm=llm, output_cls=Articles, accumulator=accumulator),
        )

        result = await engine.aquery("get an article array with title and link, link must be a valid url")
        print(result)

asyncio.run(extract_articles())