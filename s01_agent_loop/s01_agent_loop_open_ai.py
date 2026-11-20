"""
Practice: agent-loop
Paradigm-范式：REACT
    while True:
        response = LLM(messages, tools)
        if response contains no tool_use:
            break
        execute tools
        append results
"""

import json
import os
import subprocess

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

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
    f"You are a coding agent at {os.getcwd()}. Environment: {ENVIRONMENT_PROMPT}. "
    "Use bash to solve tasks. Act, don't explain."
)

# -- Tool definition: just bash -- (JSON Schema - 契约)
# OpenAI 的 tools 格式多包一层 type/function，且 schema 字段叫 parameters
TOOLS = [{
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
}]

# -- Tool execution --
def run_bash(command: str) -> str:
    dangerous = ["rm-rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, errors="replace", timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

# -- The core pattern: a while loop that calls tools until the model stops --
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
            args = json.loads(call.function.arguments or "{}")
            print("-------tool_calls_command:----------")
            print(f"\033[33m$ {args['command']}\033[0m")
            output = run_bash(args["command"])
            print("\n-------tool_calls_result:----------")
            print(f" {output[:200]} \n")
            # Feed tool result back as role=tool, loop continues
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": output,
            })

# -- Entry point --
if __name__ == "__main__":
    print("agent_loop")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    while True:
        try:
            # \001/\002 tell Readline the ANSI escapes have zero display width.
            query = input("\001\033[36m\002s01 >> \001\033[0m\002")
        except (EOFError, KeyboardInterrupt):
            break
        # 忽略空输入，不惊动模型
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

