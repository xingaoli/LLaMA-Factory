#!/usr/bin/env python3
"""Call vLLM deployed Qwen3.6 model via OpenAI-compatible API."""

import argparse
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(description="Call vLLM deployed model")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--base-url", type=str, default="http://0.0.0.0:8000/v1")
    parser.add_argument("--model", type=str, default="ckpts/Qwen3.6-27B-FP8")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-thinking", action="store_true", help="Disable thinking mode")
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": not args.no_thinking},
        },
    )
    message = response.choices[0].message
    # print(f"[Stop Thinking Mode]\n{response.choices[0].stop_reason}\n")
    print(f"[Thinking]\n{message.reasoning}\n")
    print(f"[Content]\n{message.content}")
    usage = response.usage
    print(f"[prompt_tokens]\n{usage.prompt_tokens}\n")
    print(f"[completion_tokens]\n{usage.completion_tokens}\n")

if __name__ == "__main__":
    main()
