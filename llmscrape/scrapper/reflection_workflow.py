from llama_index.core.workflow import (
    Event,
    Workflow,
    StartEvent,
    StopEvent,
    Context,
    step,
)
from llama_index.core.workflow.service import  ServiceManager
from pydantic import BaseModel
from llama_index.core.llms.utils import LLMType, resolve_llm
from typing import Optional


class ExtractionDone(Event):
    output: str
    data: str


class ValidationErrorEvent(Event):
    error: str
    wrong_output: str
    data: str


EXTRACTION_PROMPT = """
You are an json reflection expert. Always create a valid JSON object from the provided context information, and not prior knowledge.

Here are the rules you must follow strictly:
1. Answer must not contain "Here is the JSON object created from" or any similar phrase.
2. The whole response must be a valid JSON object.
3. Do not wrap the JSON object in any other structure.
4. DO NOT contain the json schema in the response.
5. DO NOT contain "$defs" in the response.

Context information is below:
---------------------
{data}
---------------------

The JSON object must follow the JSON schema:
{schema}
"""

REFLECTION_PROMPT = """
You already created this output previously:
---------------------
{wrong_answer}
---------------------

This caused the JSON decode error: {error}

Try again, the response must contain only valid JSON code. Do not add any sentence before or after the JSON object.
Do not repeat the schema.
"""


class ReflectionWorkflow(Workflow):
    def __init__(
        self,
        timeout: Optional[float] = 360.0,
        disable_validation: bool = False,
        verbose: bool = False,
        service_manager: Optional[ServiceManager] = None,
        num_concurrent_runs: Optional[int] = None,
        llm: Optional[LLMType] = None,
        max_retries: int = 3,
        output_cls = BaseModel,
    ) -> None:
        super().__init__(
            timeout=timeout,
            disable_validation=disable_validation,
            verbose=verbose,
            service_manager=service_manager,
            num_concurrent_runs=num_concurrent_runs,
        )
        self.llm = resolve_llm(llm)
        self.max_retries = max_retries
        self.output_cls = output_cls

    @step
    async def extract(
        self, ctx: Context, ev: StartEvent | ValidationErrorEvent
    ) -> StopEvent | ExtractionDone:
        current_retries = await ctx.get("retries", default=0)
        if current_retries >= self.max_retries:
            return StopEvent(result="Max retries reached")
        else:
            await ctx.set("retries", current_retries + 1)

        if isinstance(ev, StartEvent):
            data = ev.get("data")
            if not data:
                return StopEvent(result="Please provide some text in input")
            return ExtractionDone(output=data, data=data)
        elif isinstance(ev, ValidationErrorEvent):
            data = ev.data
            reflection_prompt = REFLECTION_PROMPT.format(
                wrong_answer=ev.wrong_output, error=ev.error
            )

        prompt = EXTRACTION_PROMPT.format(
            data=data, schema=self.output_cls.model_json_schema()
        )
        if reflection_prompt:
            prompt += reflection_prompt

        output = await self.llm.acomplete(prompt)

        return ExtractionDone(output=str(output), data=data)

    @step
    async def validate(
        self, ev: ExtractionDone
    ) -> StopEvent | ValidationErrorEvent:
        try:
            self.output_cls.model_validate_json(ev.output)
        except Exception as e:
            print("Validation failed, retrying...")
            return ValidationErrorEvent(
                error=str(e), wrong_output=ev.output, data=ev.data
            )

        return StopEvent(result=ev.output)