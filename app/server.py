import aiohttp, asyncio
import os

import rollbar

from io import BytesIO
from urllib.request import urlretrieve

from fastai import *
from fastai.vision import *

from telegram import ChatAction
from telegram.ext import Updater, MessageHandler, CommandHandler
from telegram.ext.filters import Filters

from dotenv import load_dotenv
load_dotenv()

# TODO
# 1. [x] Switch to env variables
# 2. [ ] Add rollbar error tracking
# 3. [ ] Setup persistent pictures storage on google cloud
# 4. [ ]

path = Path(__file__).parent
arch = models.resnet50
classes = pickle.load(open(path/'models/classes.pkl', 'rb'))

model_file_name = os.getenv('model', 'stage2-50-ep8-ep16')
size = os.getenv('size', 299)
bot_token = os.getenv('bot_token')
rollbar_token = os.getenv('rollbar_token')
rollbar_env = os.getenv('rollbar_env', 'development')


def setup_learner():
    data_bunch = ImageDataBunch.single_from_classes(path, classes,
        tfms=get_transforms(), size=size).normalize(imagenet_stats)
    learn = create_cnn(data_bunch, arch, pretrained=False)
    learn.load(model_file_name)
    return learn

def class_to_human(pred_class):
    return ' '.join(pred_class.split('-')[-1].split('_'))


def start(bot, update):
    update.message.reply_text(f"Howdy {update.message.from_user.first_name}! " +
    "Send me your doggie pic.")

def text(bot, update):
    update.message.reply_text("Please only send dog pics thx 🐕")

def photo(bot, update):
    try:
        print("      Received a photo.")

        pic = update.message.photo[-1]
        file_id = pic['file_id']
        print("      File id: " + file_id)

        print(update.message.chat)
        print(update.message.chat.id)
        bot.sendChatAction(update.message.chat.id, ChatAction.TYPING)
        print("      Sent typing notification")

        print("      Getting the image URL: ")
        # Request a link to a file that'll be valid for an hour.
        pic_url = bot.getFile(file_id)['file_path']
        print("      Done, img url: " + pic_url)

        print("      Downloading the pic to tmp...")
        pic_file_name = pic_url.split("/")[-1]
        urlretrieve(pic_url, path/'tmp'/pic_file_name)

        print("      Evaluating the image...")
        img = open_image(path/'tmp'/pic_file_name)
        pred_class, confidence, preds = learn.predict(img)
        print(f"      Breed class: {pred_class}")

        update.message.reply_text(f"It looks like a {class_to_human(pred_class)}!")

    except:
        update.message.reply_text("That was a bit too hard for me ;-(")
        rollbar.report_exc_info()


if __name__ == '__main__':
    rollbar.init(rollbar_token, rollbar_env)

    if bot_token is None:
        raise Exception("Provide bot_token env variable")

    learn = setup_learner()
    updater = Updater(bot_token)

    updater.dispatcher.add_handler(CommandHandler('start', start))
    updater.dispatcher.add_handler(MessageHandler(Filters.text, text))
    updater.dispatcher.add_handler(MessageHandler(Filters.photo, photo))

    print("Starting up...")
    updater.start_polling()
    updater.idle()
