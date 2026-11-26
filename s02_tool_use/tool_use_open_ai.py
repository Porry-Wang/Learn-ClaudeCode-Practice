"""
Practice: tool-use
adds four tools and a dispatch map
    bash
    read_file
    write_file
    edit_file
    glob
TOOL_HANDLERS instead of a hard-coded run_bash call
safe_path to keep file tools inside the workspace
"""

import json
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path(__file__).resolve().parents[1]
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)
MODEL = os.getenv("MODEL_ID", "deepseek-v4.1-flash")

ENVIRONMENT_PROMPT = (
    "Windows: the bash tool runs through cmd.exe; use cmd.exe syntax, not Unix "
    "Bash or PowerShell syntax"
    if os.name == "nt"
    else "Unix-like: the bash tool runs the system shell"
)

SYSTEM = (
    f"You are a coding agent at {WORKDIR}. Environment: {ENVIRONMENT_PROMPT}. "
    "All tool paths are relative to this root. Act, don't explain."
)


def run_bash(command: str) -> str:
    dangerous = ["rm-rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


# -- New in s02: four tools --


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path is not within the workspace: {p}")
    return path


def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit and len(lines) > limit:
            # 超出限制长度，只返回前 limit 行，并在末尾添加省略号
            lines = lines[:limit] + [f"...({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str):
    try:
        file_path = safe_path(path)
        text = file_path.read_text(encoding="utf-8")
        if old_text not in text:
            # 确保要替换的文本存在
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text), encoding="utf-8")
        return f"Replaced {old_text} with {new_text} in {path}"
    except Exception as e:
        return f"Error: {e}"


def run_glob(pattern: str) -> str:
    import glob as g

    try:
        matches = sorted(
            {
                match
                for match in g.glob(pattern, root_dir=WORKDIR, recursive=True)
                if (WORKDIR / match).resolve().is_relative_to(WORKDIR)
            }
        )
        return "\n".join(matches) if matches else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


# -- New in s02: tool definitions (one tool in s01, five in s02) --
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in a file once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern; ** matches recursively.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
]

# -- New in s02: dispatch map (replaces s01's hard-coded run_bash call) --
TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}

# s01: output = run_bash(block.input["command"])
# s02: output = TOOL_HANDLERS[block.name](**block.input)


def agent_loop(messages: list):
    for turn in range(1, 50):
        print(f"「turn {turn}」-----------------\n")
        # system 在 OpenAI 里是 messages 的第一条消息，不是顶层参数
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}, *messages],
            tools=TOOLS,
        )
        msg = response.choices[0].message

        # Append assistant turn (tool_calls 必须原样带回，否则下一轮 400)
        messages.append(msg.model_dump(exclude_none=True))

        # If the model didn't call a tool, we're done
        tool_calls = msg.tool_calls or []
        if not tool_calls:
            return

        # Execute each tool call; OpenAI 的入参是 JSON 字符串，不是现成的 dict
        for call in tool_calls:
            function = call.function
            print(f"-------function: {function} ----------")
            args = json.loads(function.arguments or "{}")
            print("-------tool_calls_args:----------")
            print(f"\033[33m$ {args}\033[0m")
            handler = TOOL_HANDLERS.get(function.name)
            try:
                call_results = (
                    handler(**args)
                    if handler
                    else f"Error: Tool {function.name} not found"
                )
            except Exception as e:
                # 工具异常必须隔离在单个 call 内，否则 tool_calls 缺配对会让下轮请求 400
                call_results = f"Error: tool {function.name} failed: {e}"
            print("\n-------tool_calls_result:----------")
            print(f" {str(call_results)[:200]} \n")
            # Feed tool result back as role=tool, loop continues
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": call_results,
                }
            )


if __name__ == "__main__":
    print("s02: Tool Use - four tools added to s01")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    while True:
        try:
            # \001/\002 tell Readline the ANSI escapes have zero display width.
            query = input("\001\033[36m\002s02 >> \001\033[0m\002")
        except (EOFError, KeyboardInterrupt):
            break
        if not query.strip():
            continue
        if query.strip().lower() in ("q", "exit"):
            break

        history.append({"role": "user", "content": query})
        agent_loop(history)
        # Print the model's final text response (OpenAI 下就是普通字符串)
        final = history[-1].get("content")
        if isinstance(final, str):
            print("-------final----------\n")
            print(f" {final} \n")
            print("------------------\n")
        print()
