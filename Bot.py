import logging
import asyncio
import torch
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import Message
from aiogram.filters import Command
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

TOKEN = "7471672611:AAGXtQnxTPQ8gla5O9ME4Uc-j5fnu1s5vqo"

logging.basicConfig(level=logging.INFO)

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

bot = Bot(token=TOKEN)
dp = Dispatcher(bot=bot)

@dp.message(Command("start"))
async def start(message: Message):
    await message.answer("👋 Hello! Send me a photo, and I'll describe it in English! 📸")

@dp.message(F.photo)
async def handle_photo(message: Message):
    await message.answer("🔍 Processing image...")

    try:
        photo = message.photo[-1]
        photo_file = await bot.download(photo.file_id)

        image = Image.open(photo_file)

        inputs = processor(image, text="a picture of", return_tensors="pt").to(device)
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

        await message.answer(f"📸 Description: {caption}")

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        await message.answer("❌ An error occurred while processing the photo.")

async def main():
    await bot.delete_webhook(drop_pending_updates=True)  # Deletes webhook before polling
    await dp.start_polling(bot, skip_updates=True)      # Starts polling

if __name__ == "__main__":
    asyncio.run(main())