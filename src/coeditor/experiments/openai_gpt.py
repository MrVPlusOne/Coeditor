import openai
import tiktoken

from coeditor.common import *
from coeditor.encoding import TruncateAt, truncate_sections


@dataclass
class OpenAIGptWrapper:
    model: str = "gpt-3.5-turbo"
    tks_limit: int = 4096
    use_fim: bool = False
    print_prompt: bool = False

    def __post_init__(self):
        api_key = openai.api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise RuntimeError("OPENAI_API_KEY env variable not set.")
        self.tokenizer = tiktoken.encoding_for_model(self.model)

    def infill(self, left: str, right: str, max_output: int) -> str:
        if self.use_fim:
            return self.infill_fim(left, right, max_output)
        else:
            return self.infill_lm(left, max_output)

    def infill_lm(self, left: str, max_output: int) -> str:
        """Infill code using left-to-right language modeling."""
        left_tks = self.tokenizer.encode(left)
        left_tks = truncate_sections(
            self.tks_limit - max_output - 500,
            (left_tks, TruncateAt.Left),
            add_bos=False,
        )[0]
        left_str = self.tokenizer.decode(left_tks)
        #         prompt = f"""\
        # Output the next line of the following Python code snippet.
        # ---
        # {left_str}
        # """
        prompt = left_str
        self._print_prompt(prompt)
        return self._get_result(prompt, role="assistant", max_output=max_output)

    def infill_fim(self, left: str, right: str, max_output: int) -> str:
        """Infill code using FIM prompting."""

        prompt = """\
You are tasked to fill in a missing line for a given Python code snippet (which
may have been truncated).
The missing line is indicated by `<MISSING LINE>`. You should
then output the missing line (including the leading whitespaces) and nothing more.
For example, if the input is
```
def fib(n):
    if n < 2:
MISSING LINE
    else:
        return fib(n-1) + fib(
```
You should output `        return 1` and nothing more (without the quote).
Now complete the code below:
```
{}<MISSING LINE>{}
```
"""
        prompt_len = len(self.tokenizer.encode(prompt))
        left_tks = self.tokenizer.encode(left)
        right_tks = self.tokenizer.encode(right)
        left_tks, right_tks = truncate_sections(
            self.tks_limit - max_output - 400 - prompt_len,
            (left_tks, TruncateAt.Left),
            (right_tks, TruncateAt.Right),
            add_bos=False,
        )
        left_str = self.tokenizer.decode(left_tks)
        right_str = self.tokenizer.decode(right_tks)
        prompt = prompt.format(left_str, right_str)
        self._print_prompt(prompt)
        return self._get_result(prompt, max_output=max_output)

    def _get_result(self, prompt: str, max_output: int, role: str = "user") -> str:
        messages = [{"role": role, "content": prompt}]
        completion: Any = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_output,
            stop=["\n"],
        )
        result = completion.choices[0].message.content
        assert isinstance(result, str)
        return result

    def _print_prompt(self, prompt: str):
        if self.print_prompt:
            print(prompt)
            print(SEP)
            print("End of Prompt")
            print(SEP)


my_password = "password123"
