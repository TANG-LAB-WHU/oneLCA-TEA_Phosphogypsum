import os
import pandas as pd
import asyncio
import aiohttp
from dotenv import load_dotenv
from googletrans import Translator

# Load environment variables
load_dotenv()

# Define the folder containing the Excel files
folder_path = "c:\\Github\\oneLCA-TEA_Phosphogypsum\\data\\raw\\agribalyse"

# Gemini API configuration
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")

# Parameter to control translation mode
USE_GEMINI_API = os.getenv("USE_GEMINI_API", "false").lower() == "true"

# Function to translate text using Gemini API
async def translate_text_gemini(session, text, src_lang="fr", dest_lang="en"):
    url = f"{LLM_BASE_URL}/translate"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {"text": text, "source_language": src_lang, "target_language": dest_lang}

    async with session.post(url, json=payload, headers=headers) as response:
        if response.status == 200:
            result = await response.json()
            return result.get("translated_text", text)
        else:
            print(f"Translation failed: {response.status}, {await response.text()}")
            return text

# Function to translate text using googletrans
def translate_text_googletrans(translator, text, src_lang="fr", dest_lang="en"):
    try:
        translated = translator.translate(text, src=src_lang, dest=dest_lang)
        return translated.text
    except Exception as e:
        print(f"Googletrans translation failed: {e}")
        return text

# Function to translate a DataFrame
async def translate_dataframe(df):
    translated_df = df.copy()

    if USE_GEMINI_API:
        async with aiohttp.ClientSession() as session:
            # Translate column names
            column_tasks = [translate_text_gemini(session, col) for col in df.columns]
            translated_columns = await asyncio.gather(*column_tasks)
            translated_df.columns = translated_columns

            # Translate cell values if they are strings
            for col in df.columns:
                if df[col].dtype == 'object':
                    cell_tasks = [translate_text_gemini(session, str(value)) if isinstance(value, str) else value for value in df[col]]
                    translated_df[col] = await asyncio.gather(*cell_tasks)
    else:
        translator = Translator()
        # Translate column names
        translated_columns = [translate_text_googletrans(translator, col) for col in df.columns]
        translated_df.columns = translated_columns

        # Translate cell values if they are strings
        for col in df.columns:
            if df[col].dtype == 'object':
                translated_df[col] = df[col].apply(lambda x: translate_text_googletrans(translator, x) if isinstance(x, str) else x)

    return translated_df

# Main function to process files
async def main():
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".xlsx"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_name}")

            # Read the Excel file
            try:
                df = pd.read_excel(file_path)
            except Exception as e:
                print(f"Failed to read {file_name}: {e}")
                continue

            # Translate the DataFrame
            try:
                translated_df = await translate_dataframe(df)
            except Exception as e:
                print(f"Failed to translate {file_name}: {e}")
                continue

            # Save the translated DataFrame to a new Excel file
            translated_file_name = f"translated_{file_name}"
            translated_file_path = os.path.join(folder_path, translated_file_name)
            try:
                translated_df.to_excel(translated_file_path, index=False)
                print(f"Translated file saved as: {translated_file_name}")
            except Exception as e:
                print(f"Failed to save translated file for {file_name}: {e}")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())