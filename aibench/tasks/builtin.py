"""Built-in task suites.

Chosen to (a) exercise different generation profiles and (b) *discriminate* —
mixing easy items with hard ones that weaker models fail, so quality scores
don't all pin at 1.0. Code tasks are executed against hidden edge cases;
several creative tasks use objectively checkable constraints (exact length,
acrostic) so they discriminate without a judge model.
"""

from __future__ import annotations

from .base import Task, TaskSuite

_ARTICLE = (
    "Large language models are increasingly run on local hardware rather than "
    "in the cloud. Running a model locally keeps data on the device, removes "
    "per-token API costs, and lets teams tune latency by choosing their own "
    "hardware. The trade-off is that a single machine has fixed memory and "
    "compute, so operators must weigh model size against speed. Quantization "
    "shrinks weights to fit larger models into limited VRAM at some cost to "
    "quality, while batching improves throughput when many requests arrive at "
    "once. Benchmarking on representative tasks is the only reliable way to "
    "know whether a given model and machine meet an application's needs."
)

_MEETING = (
    "Alright team. Priya, please finalize the Q3 budget by Friday and send it "
    "to Marcus. Marcus, once you have it, book the venue for the offsite. "
    "Dana will draft the customer email and share it for review on Thursday. "
    "We also agreed to postpone the logo redesign until next quarter."
)

# --- creative writing -----------------------------------------------------

CREATIVE_WRITING = TaskSuite(
    name="creative_writing",
    description="Free-form generation; some objectively-checkable constraints.",
    tasks=[
        Task(
            id="cw_short_story", category="creative_writing", difficulty="easy",
            prompt=(
                "Write a 400-word short story about a lighthouse keeper who "
                "discovers the light is talking back. Give it a clear beginning, "
                "middle, and end."
            ),
            params={"max_tokens": 700, "temperature": 0.9},
            checks=[{"type": "word_count", "min": 250, "max": 600}],
        ),
        Task(
            id="cw_poem", category="creative_writing", difficulty="easy",
            prompt=(
                "Write a vivid four-stanza poem about the first snowfall in a "
                "quiet mountain town."
            ),
            params={"max_tokens": 400, "temperature": 0.9},
            checks=[{"type": "stanza_count", "n": 4}],
        ),
        Task(
            id="cw_six_word", category="creative_writing", difficulty="hard",
            system="Follow the length constraint exactly. Output only the story.",
            prompt="Write a six-word story about saying goodbye. Exactly six words.",
            params={"max_tokens": 60, "temperature": 0.8},
            checks=[{"type": "word_count", "min": 6, "max": 6}],
        ),
        Task(
            id="cw_acrostic", category="creative_writing", difficulty="hard",
            system="Output only the poem, one line per letter.",
            prompt=(
                "Write an acrostic poem where the first letters of the lines "
                "spell OCEAN (one line each for O, C, E, A, N)."
            ),
            params={"max_tokens": 200, "temperature": 0.7},
            checks=[{"type": "acrostic", "word": "OCEAN"}],
        ),
    ],
)

# --- code generation (executed) -------------------------------------------

CODE_GENERATION = TaskSuite(
    name="code_generation",
    description="Code executed against hidden cases; SQL run on a real DB.",
    tasks=[
        Task(
            id="code_fib", category="code_generation", difficulty="easy",
            system="You are a precise coding assistant. Return only code.",
            prompt=(
                "Write a Python function `fib(n)` that returns the nth Fibonacci "
                "number iteratively in O(n) time, with a docstring and type hints."
            ),
            params={"max_tokens": 500, "temperature": 0.2},
            checks=[{"type": "python_func", "func": "fib", "cases": [
                {"args": [0], "expect": 0}, {"args": [1], "expect": 1},
                {"args": [2], "expect": 1}, {"args": [5], "expect": 5},
                {"args": [10], "expect": 55}, {"args": [20], "expect": 6765},
            ]}],
        ),
        Task(
            id="code_valid_parens", category="code_generation", difficulty="medium",
            system="You are a precise coding assistant. Return only code.",
            prompt=(
                "Write a Python function `is_valid(s: str) -> bool` that returns "
                "True iff the brackets in s are balanced and correctly nested. "
                "Consider (), [] and {}."
            ),
            params={"max_tokens": 500, "temperature": 0.2},
            checks=[{"type": "python_func", "func": "is_valid", "cases": [
                {"args": ["()"], "expect": True},
                {"args": ["()[]{}"], "expect": True},
                {"args": ["(]"], "expect": False},
                {"args": ["([)]"], "expect": False},
                {"args": ["{[]}"], "expect": True},
                {"args": [""], "expect": True},
                {"args": ["("], "expect": False},
            ]}],
        ),
        Task(
            id="code_roman", category="code_generation", difficulty="medium",
            system="You are a precise coding assistant. Return only code.",
            prompt=(
                "Write a Python function `roman_to_int(s: str) -> int` that "
                "converts a valid Roman numeral string to an integer."
            ),
            params={"max_tokens": 600, "temperature": 0.2},
            checks=[{"type": "python_func", "func": "roman_to_int", "cases": [
                {"args": ["III"], "expect": 3}, {"args": ["IV"], "expect": 4},
                {"args": ["IX"], "expect": 9}, {"args": ["LVIII"], "expect": 58},
                {"args": ["MCMXCIV"], "expect": 1994},
                {"args": ["MMXXIV"], "expect": 2024},
            ]}],
        ),
        Task(
            id="code_merge_intervals", category="code_generation", difficulty="hard",
            system="You are a precise coding assistant. Return only code.",
            prompt=(
                "Write a Python function `merge(intervals: list[list[int]]) -> "
                "list[list[int]]` that merges all overlapping intervals and "
                "returns them sorted by start."
            ),
            params={"max_tokens": 700, "temperature": 0.2},
            checks=[{"type": "python_func", "func": "merge", "cases": [
                {"args": [[[1, 3], [2, 6], [8, 10], [15, 18]]],
                 "expect": [[1, 6], [8, 10], [15, 18]]},
                {"args": [[[1, 4], [4, 5]]], "expect": [[1, 5]]},
                {"args": [[[1, 4], [0, 4]]], "expect": [[0, 4]]},
                {"args": [[[1, 4], [2, 3]]], "expect": [[1, 4]]},
                {"args": [[[1, 4]]], "expect": [[1, 4]]},
            ]}],
        ),
        Task(
            id="code_edit_distance", category="code_generation", difficulty="hard",
            system="You are a precise coding assistant. Return only code.",
            prompt=(
                "Write a Python function `min_distance(a: str, b: str) -> int` "
                "returning the Levenshtein edit distance between a and b "
                "(insert/delete/replace, each cost 1)."
            ),
            params={"max_tokens": 700, "temperature": 0.2},
            checks=[{"type": "python_func", "func": "min_distance", "cases": [
                {"args": ["horse", "ros"], "expect": 3},
                {"args": ["intention", "execution"], "expect": 5},
                {"args": ["", "abc"], "expect": 3},
                {"args": ["abc", "abc"], "expect": 0},
                {"args": ["kitten", "sitting"], "expect": 3},
            ]}],
        ),
        Task(
            id="code_sql", category="code_generation", difficulty="medium",
            system="You are a precise coding assistant.",
            prompt=(
                "Given tables `orders(id, customer_id, total, created_at)` and "
                "`customers(id, name)`, write a SQL query returning the top 5 "
                "customers by total spend in 2024, with their name and total."
            ),
            params={"max_tokens": 400, "temperature": 0.2},
            checks=[{
                "type": "sql_sqlite",
                "customer_names": ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"],
                "expect_order": ["Dave", "Bob", "Frank", "Alice", "Carol"],
                "setup": """
                    CREATE TABLE customers (id INTEGER, name TEXT);
                    INSERT INTO customers VALUES
                      (1,'Alice'),(2,'Bob'),(3,'Carol'),
                      (4,'Dave'),(5,'Eve'),(6,'Frank');
                    CREATE TABLE orders (id INTEGER, customer_id INTEGER,
                                         total REAL, created_at TEXT);
                    INSERT INTO orders VALUES
                      (1,1,500,'2024-03-10'),(2,1,300,'2024-07-01'),
                      (3,2,1000,'2024-05-05'),(4,3,250,'2024-02-20'),
                      (5,4,700,'2024-01-15'),(6,4,700,'2024-11-30'),
                      (7,5,50,'2024-06-06'),(8,6,900,'2024-09-09'),
                      (9,1,9999,'2023-12-31');
                """,
            }],
        ),
    ],
)

# --- summarization --------------------------------------------------------

SUMMARIZATION = TaskSuite(
    name="summarization",
    description="Read-heavy, short output; format and key-concept coverage.",
    tasks=[
        Task(
            id="sum_article", category="summarization", difficulty="easy",
            prompt=(
                "Summarize the following text in exactly three bullet points, "
                "each under 20 words:\n\n" + _ARTICLE
            ),
            params={"max_tokens": 250, "temperature": 0.3},
            reference=_ARTICLE,
            checks=[
                {"type": "bullets", "n": 3, "max_words": 20},
                {"type": "keyword_coverage", "keywords": [
                    ["privacy", "on the device", "on device", "data stays"],
                    ["cost", "api costs", "per-token"],
                    "quantization", ["batching", "throughput"], "benchmark",
                ]},
            ],
        ),
        Task(
            id="sum_tldr", category="summarization", difficulty="medium",
            prompt=(
                "Give a single-sentence TL;DR (one sentence only) of:\n\n"
                + _ARTICLE
            ),
            params={"max_tokens": 120, "temperature": 0.3},
            checks=[
                {"type": "sentence_count", "n": 1, "weight": 1},
                {"type": "keyword_coverage", "weight": 1, "keywords": [
                    ["local", "on device"], ["hardware", "vram", "memory"],
                    ["benchmark", "trade-off", "speed"],
                ]},
            ],
        ),
        Task(
            id="sum_action_items", category="summarization", difficulty="hard",
            system="Extract only the action items as bullet points.",
            prompt=(
                "List the action items from this meeting transcript as bullet "
                "points, one per task, naming who owns each:\n\n" + _MEETING
            ),
            params={"max_tokens": 250, "temperature": 0.3},
            checks=[
                {"type": "bullets", "n": 3, "weight": 1},
                {"type": "keyword_coverage", "weight": 2, "keywords": [
                    "priya", "marcus", "dana", ["budget"], ["venue", "offsite"],
                    ["email"],
                ]},
            ],
        ),
    ],
)

# --- tool use -------------------------------------------------------------

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name."}},
            "required": ["city"],
        },
    },
}
_CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}
_STOCK_TOOL = {
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Get the latest stock price for a ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
}
_TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current time in a timezone.",
        "parameters": {
            "type": "object",
            "properties": {"timezone": {"type": "string"}},
            "required": ["timezone"],
        },
    },
}

TOOL_USE = TaskSuite(
    name="tool_use",
    description="Agent/tool-use: correct call, arg accuracy, tool selection, restraint.",
    tasks=[
        Task(
            id="tool_weather", category="tool_use", difficulty="easy",
            system="You have tools. Call one when it answers the question.",
            prompt="What's the weather in Tokyo right now? Use the tool.",
            params={"max_tokens": 200, "temperature": 0.0},
            expects_tool="get_weather", tools=[_WEATHER_TOOL],
            checks=[{"type": "tool_call", "name": "get_weather",
                     "args_include": {"city": "Tokyo"}}],
        ),
        Task(
            id="tool_calc", category="tool_use", difficulty="medium",
            system="You have tools. Use the calculator for arithmetic.",
            prompt="What is 47 multiplied by 89? Use the calculator tool.",
            params={"max_tokens": 200, "temperature": 0.0},
            tools=[_CALC_TOOL],
            checks=[{"type": "tool_call", "name": "calculate"}],
        ),
        Task(
            id="tool_select", category="tool_use", difficulty="hard",
            system="You have several tools. Pick the right one.",
            prompt="What's the latest stock price of Apple (AAPL)?",
            params={"max_tokens": 200, "temperature": 0.0},
            tools=[_WEATHER_TOOL, _STOCK_TOOL, _TIME_TOOL],
            checks=[{"type": "tool_call", "name": "get_stock_price",
                     "args_include": {"ticker": "AAPL"}}],
        ),
        Task(
            id="tool_restraint", category="tool_use", difficulty="hard",
            system="You have tools, but only use one if it is actually needed.",
            prompt="In one sentence, explain what photosynthesis is.",
            params={"max_tokens": 200, "temperature": 0.0},
            tools=[_WEATHER_TOOL, _STOCK_TOOL],
            checks=[{"type": "tool_call", "expect_no_call": True}],
        ),
    ],
)

ALL_SUITES: dict[str, TaskSuite] = {
    s.name: s
    for s in (CREATIVE_WRITING, CODE_GENERATION, SUMMARIZATION, TOOL_USE)
}


def resolve_suites(names: list[str]) -> list[TaskSuite]:
    """Resolve suite names to suites. `["all"]` (or empty) returns every suite."""
    if not names or names == ["all"]:
        return list(ALL_SUITES.values())
    out: list[TaskSuite] = []
    unknown: list[str] = []
    for n in names:
        if n in ALL_SUITES:
            out.append(ALL_SUITES[n])
        else:
            unknown.append(n)
    if unknown:
        raise ValueError(
            f"Unknown task suite(s): {', '.join(unknown)}. "
            f"Available: {', '.join(ALL_SUITES)}"
        )
    return out
