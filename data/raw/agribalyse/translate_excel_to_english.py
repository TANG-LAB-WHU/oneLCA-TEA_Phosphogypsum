"""
Translate Agribalyse Excel files from French to English.

What this script does:
- Reads all `.xlsx` files from a target folder.
- Translates column names and string cell values to English.
- Writes translated files as `translated_<original_name>.xlsx` to an output folder.

Translator backend priority:
1) CLI argument `--translator` (highest priority)
2) Environment variable `USE_LLM_API=true|false` (fallback)
3) Default backend: `googletrans`

Usage examples:
- Use OpenAI-compatible LLM backend (e.g. Ollama):
  python translate_excel_to_english.py --translator llm

- Use googletrans backend:
  python translate_excel_to_english.py --translator googletrans

- Use custom folder:
  python translate_excel_to_english.py --translator llm --input-folder .

- Use custom output folder:
  python translate_excel_to_english.py --translator llm --output-dir output

- Use safer LLM settings under rate limit pressure:
  python translate_excel_to_english.py --translator llm --max-concurrency 1 --max-retries 6

Notes:
- `--input-folder` defaults to the script directory.
- `--output-dir` defaults to the script directory.
- Files prefixed with `translated_` are skipped automatically.
- LLM mode uses LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL from the environment.
"""

import argparse
import asyncio
import os

import aiohttp
import pandas as pd
from dotenv import load_dotenv
from googletrans import Translator

# Load environment variables
load_dotenv()

# Default folder containing the Excel files (script directory)
DEFAULT_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

# OpenAI-compatible API (e.g. Ollama /v1)
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:35b")
DEFAULT_MAX_CONCURRENCY = 3
DEFAULT_MAX_RETRIES = 4
DEFAULT_BACKOFF_BASE_SECONDS = 1.5


def parse_args():
    parser = argparse.ArgumentParser(description="Translate Agribalyse Excel files to English")
    parser.add_argument(
        "--translator",
        choices=["llm", "googletrans"],
        default=None,
        help="Translation backend to use (CLI takes priority over environment variable)",
    )
    parser.add_argument(
        "--input-folder",
        dest="input_folder",
        default=None,
        help="Folder containing source .xlsx files (CLI takes priority over default path)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder to save translated .xlsx files (default: script directory)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help=f"Maximum concurrent LLM requests (default: {DEFAULT_MAX_CONCURRENCY})",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum retries for transient LLM errors (default: {DEFAULT_MAX_RETRIES})",
    )
    return parser.parse_args()


def resolve_translator(cli_translator):
    if cli_translator:
        return cli_translator
    return "llm" if os.getenv("USE_LLM_API", "false").lower() == "true" else "googletrans"


async def _translate_text_llm_request(session, text, src_lang="fr", dest_lang="en"):
    base = (LLM_BASE_URL or "http://127.0.0.1:11434/v1").rstrip("/")
    url = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY or 'ollama'}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You are a translation engine. Translate from {src_lang} to {dest_lang}. "
                    "Return only the translated text without explanation."
                ),
            },
            {"role": "user", "content": text},
        ],
        "temperature": 0,
    }

    async with session.post(url, json=payload, headers=headers) as response:
        response_text = await response.text()
        if response.status == 200:
            try:
                result = await response.json()
                translated = result["choices"][0]["message"]["content"]
                return translated.strip() if translated else text
            except (KeyError, IndexError, TypeError, aiohttp.ContentTypeError):
                print(f"Unexpected response schema: {response_text[:300]}")
                return text

        if response.status in {429, 500, 502, 503, 504}:
            raise RuntimeError(
                f"Transient translation error {response.status}: {response_text[:300]}"
            )

        print(f"Translation failed: {response.status}, {response_text[:300]}")
        return text


async def translate_text_llm(
    session,
    semaphore,
    text,
    src_lang="fr",
    dest_lang="en",
    max_retries=DEFAULT_MAX_RETRIES,
):
    if not text:
        return text

    for attempt in range(max_retries + 1):
        try:
            async with semaphore:
                return await _translate_text_llm_request(
                    session, text, src_lang=src_lang, dest_lang=dest_lang
                )
        except Exception as exc:
            if attempt >= max_retries:
                print(f"Translation failed after retries: {exc}")
                return text
            backoff_seconds = DEFAULT_BACKOFF_BASE_SECONDS * (2**attempt)
            await asyncio.sleep(backoff_seconds)


def translate_text_googletrans(translator, text, src_lang="fr", dest_lang="en"):
    try:
        translated = translator.translate(text, src=src_lang, dest=dest_lang)
        return translated.text
    except Exception as e:
        print(f"Googletrans translation failed: {e}")
        return text


async def translate_dataframe(
    df,
    use_llm_api=False,
    max_concurrency=DEFAULT_MAX_CONCURRENCY,
    max_retries=DEFAULT_MAX_RETRIES,
):
    translated_df = df.copy()

    if use_llm_api:
        semaphore = asyncio.Semaphore(max_concurrency)
        async with aiohttp.ClientSession() as session:
            column_tasks = [
                translate_text_llm(
                    session,
                    semaphore,
                    str(col),
                    max_retries=max_retries,
                )
                for col in df.columns
            ]
            translated_columns = await asyncio.gather(*column_tasks)
            translated_df.columns = translated_columns

            for col in df.columns:
                if df[col].dtype == "object":
                    cell_tasks = [
                        (
                            translate_text_llm(
                                session,
                                semaphore,
                                str(value),
                                max_retries=max_retries,
                            )
                            if isinstance(value, str)
                            else value
                        )
                        for value in df[col]
                    ]
                    translated_df[col] = await asyncio.gather(*cell_tasks)
    else:
        translator = Translator()
        translated_columns = [translate_text_googletrans(translator, col) for col in df.columns]
        translated_df.columns = translated_columns

        for col in df.columns:
            if df[col].dtype == "object":
                translated_df[col] = df[col].apply(
                    lambda x: translate_text_googletrans(translator, x) if isinstance(x, str) else x
                )

    return translated_df


async def _main_async(folder_path, output_dir, use_llm_api, max_concurrency, max_retries):
    os.makedirs(output_dir, exist_ok=True)
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".xlsx"):
            if file_name.startswith("translated_"):
                print(f"Skipping already translated file: {file_name}")
                continue
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_name}")

            try:
                df = pd.read_excel(file_path)
            except Exception as e:
                print(f"Failed to read {file_name}: {e}")
                continue

            try:
                translated_df = await translate_dataframe(
                    df,
                    use_llm_api=use_llm_api,
                    max_concurrency=max_concurrency,
                    max_retries=max_retries,
                )
            except Exception as e:
                print(f"Failed to translate {file_name}: {e}")
                continue

            translated_file_name = f"translated_{file_name}"
            translated_file_path = os.path.join(output_dir, translated_file_name)
            try:
                translated_df.to_excel(translated_file_path, index=False)
                print(f"Translated file saved as: {translated_file_name}")
            except Exception as e:
                print(f"Failed to save translated file for {file_name}: {e}")


def main():
    args = parse_args()
    selected_translator = resolve_translator(args.translator)
    use_llm_api = selected_translator == "llm"
    target_folder = args.input_folder if args.input_folder else DEFAULT_FOLDER_PATH
    output_dir = args.output_dir if args.output_dir else DEFAULT_FOLDER_PATH
    max_concurrency = max(1, args.max_concurrency)
    max_retries = max(0, args.max_retries)

    print(f"Translator backend: {selected_translator}")
    print(f"Target folder: {target_folder}")
    print(f"Output folder: {output_dir}")
    print(f"Max concurrency: {max_concurrency}")
    print(f"Max retries: {max_retries}")
    asyncio.run(
        _main_async(target_folder, output_dir, use_llm_api, max_concurrency, max_retries)
    )


if __name__ == "__main__":
    main()
