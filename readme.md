# Dog breeds classification in a telegram bot

This is a demo project that shows off how to build and serve a convolutional neural net classifier in production. 

The code is not production ready by any means, but feel free to play around with it. 

Hit up [@whatdoggiebot](https://t.me/whatdoggiebot) on Telegram. 


## Installation

You can easily run your own classifier model with the same code in a few steps: 

1. Create an `.env` (see example in `.env.sample`), provide a Rollbar API token and a TG bot token. 
2. Put your trained model `.pth` file in app/models and set the model name and pics size on `.env`. Don't forget to tweak `arch` in `server.py` as well. 
3. `pip install -r requirements.txt`
4. `python app/server.py` should start the bot and print `Starting up...` to your terminal.

Please see [Telegram docs](https://core.telegram.org/bots) on how to create bots and what the bots API can do.


## Training and running your model 

This classifier was build with [fast.ai](https://fast.ai) [v1](https://github.com/fastai/fastai). They have MOOCs on deep learning. The whole learning process with downloading the dataset took a few hours, but less than a day. 

The model runs on CPU for inference in production, you don't need a GPU to run it.


## Deploying the model

Whatdoggiebot runs on a single Google Cloud VM, deployed as a docker image via Google Cloud Registry. 

```
docker build . -t gcr.io/PROJECT_ID/whatdoggie:latest
docker push gcr.io/PROJECT_ID/whatdoggie:latest
gcloud compute instances reset whatdoggie-vm
```

The VM is ~$10 monthly, but any free docker hosting will work as a free alternative.


## Data and inspiration

This model was trained on the Stanford dog breeds dataset and works pretty crappy with real world dog pics ;-). If you want to fool around, you can build your own dataset with just google image search, or use a subset of [Google Open Images](https://storage.googleapis.com/openimages/web/index.html). 

Have fun! 

Let me know what you built [on twitter](https://twitter.com/xnutsive)! 