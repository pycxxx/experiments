import asyncio
import json
import logging
from typing import Any, List, Optional, Sequence, Type, Callable
from llama_index.core.async_utils import run_async_tasks
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.llms import LLM
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.types import RESPONSE_TEXT_TYPE
from llama_index.core import ChatPromptTemplate
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from .reflection_workflow import ReflectionWorkflow

logger = logging.getLogger(__name__)

TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "You are an expert web content scrapper that is trusted around the world.\n"
        "Always extract the content from the provided context information, "
        "and not prior knowledge.\n"
        "The provided context information is a web page content in markdown format.\n"
        "Some rules to follow:\n"
        "1. Answer should directly extracted from the context information.\n"
        "2. Avoid statements like 'Based on the context, ...' or "
        "'The context information ...' or anything along "
        "those lines. \n"
        "3. Do not contain the schema in the response."
    ),
    role=MessageRole.SYSTEM,
)

TEXT_QA_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "answer must stricly follow the JSON schema:\n"
            "---------------------\n"
            "{schema}\n"
            "---------------------\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]

CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)


class StructuredAccumulate(BaseSynthesizer):
    """StructuredAccumulate responses from multiple text chunks."""

    def __init__(
        self,
        accumulator: Callable[[BaseModel | None, BaseModel], BaseModel],
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
        prompt_helper: Optional[PromptHelper] = None,
        text_qa_template: Optional[BasePromptTemplate] = None,
        output_cls: Type[BaseModel] = None,
        streaming: bool = False,
        use_async: bool = False,
    ) -> None:
        super().__init__(
            llm=llm,
            callback_manager=callback_manager,
            prompt_helper=prompt_helper,
            streaming=streaming,
        )
        self._text_qa_template = text_qa_template or CHAT_TEXT_QA_PROMPT
        self._use_async = use_async
        self._output_cls = output_cls
        self._accumulator = accumulator

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"text_qa_template": self._text_qa_template}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "text_qa_template" in prompts:
            self._text_qa_template = prompts["text_qa_template"]

    def flatten_list(self, md_array: List[List[Any]]) -> List[Any]:
        return [item for sublist in md_array for item in sublist]

    def _merge_outputs(self, outputs: List[BaseModel | None]) -> str:
        """Merge outputs."""
        acc: BaseModel | None = None
        for o in outputs:
            if o is not None:
                acc = self._accumulator(acc, o)
        if acc is None:
            return ""
        return acc.model_dump_json()

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Apply the same prompt to text chunks and return async responses."""
        if self._streaming:
            raise ValueError("Unable to stream in Accumulate response mode")

        tasks = [
            self._give_responses(
                query_str, text_chunk, use_async=True, **response_kwargs
            )
            for text_chunk in text_chunks
        ]

        flattened_tasks = self.flatten_list(tasks)
        outputs = await asyncio.gather(*flattened_tasks)

        return self._merge_outputs(outputs)

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Apply the same prompt to text chunks and return responses."""
        if self._streaming:
            raise ValueError("Unable to stream in Accumulate response mode")

        tasks = [
            self._give_responses(
                query_str, text_chunk, use_async=self._use_async, **response_kwargs
            )
            for text_chunk in text_chunks
        ]

        outputs = self.flatten_list(tasks)

        if self._use_async:
            outputs = run_async_tasks(outputs)

        return self._merge_outputs(outputs)

    def _give_responses(
        self,
        query_str: str,
        text_chunk: str,
        use_async: bool = False,
        **response_kwargs: Any,
    ) -> List[Any]:
        """Give responses given a query and a corresponding text chunk."""
        text_qa_template = self._text_qa_template.partial_format(
            query_str=query_str,
            schema=self._output_cls.model_json_schema(),
        )

        text_chunks = self._prompt_helper.repack(
            text_qa_template, [text_chunk], llm=self._llm
        )

        return [
            self._give_response(
                text_qa_template=text_qa_template,
                text_chunk=cur_text_chunk,
                use_async=use_async,
                **response_kwargs,
            )
            for cur_text_chunk in text_chunks
        ]

    async def _give_response(
        self,
        text_chunk: str,
        text_qa_template: ChatPromptTemplate,
        use_async: bool = False,
        **response_kwargs: Any,
    ) -> Any:
        """Give response given a query and a corresponding text chunk."""

        predictor = self._llm.apredict if use_async else self._llm.predict
        result = await predictor(
            text_qa_template,
            context_str=text_chunk,
            **response_kwargs,
        )

        try:
            return self._output_cls.model_validate_json(result)
        except Exception as e:
            logger.debug(f"> failed to parse llm response ({e}), run the reflection workflow")

        reflection_workflow = ReflectionWorkflow(llm=self._llm, output_cls=self._output_cls, verbose=True)
        structured_result = await reflection_workflow.run(data=result)
        try:
            return self._output_cls.model_validate_json(structured_result)
        except Exception as e:
            logger.debug(f"> failed to parse llm response: {structured_result}")
            return None