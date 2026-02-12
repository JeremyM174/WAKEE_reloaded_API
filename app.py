import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from PIL import Image
from io import BytesIO
import cnn
import llm
from dotenv import load_dotenv
load_dotenv()



description = """
**Welcome to our image analysis with WAKEE.reloaded!**
![WAKEE](https://i83.servimg.com/u/f83/17/01/47/37/wakee_10.png)

This short documentation is meant to help you understand how to use our API, so our model can evaluate whether you're experimenting cognitive drift.

Please remember this is merely a proof of concept for a student project; it is **by no way** meant to be used any other way!

---

### Root

* `/`: **GET** request producing the html content of the website.

### Test

* `/test`: **POST** request returning what it is being sent (or "dumb mirror"), so you can verify what the API receives. 5 calls/minute.

### Predict

* `/predict`: **POST** request accepting an image as input, returning a personalized recommendation when cognitive drift is recognized. 2 calls/minute.

### Backup

* `/backup`: **POST** request similar to /predict, minus the personalized recommendation. Use when /predict has hit its limit, or after any error message at all. 2 calls/minute.

---

This API has limits over its endpoints to avoid overloading it, with the generator of personalized recommendations behind WAKEE also having usage limits of its own.

The /backup endpoint does not call that generator and can be a solution if /predict doesn't work. Otherwise beyond the endpoints' intended use, your requests will return an error message!

You may find below request examples towards our /predict endpoint depending your operating system:

**Linux/Mac**
```curl -X POST https://mevelios-wakee-reloaded-api.hf.space/predict -H "Content-Type: image/jpg" --data-binary @frame175751278672.jpg```

**Windows**
```curl.exe -X POST https://mevelios-wakee-reloaded-api.hf.space/predict -H "Content-Type: image/jpeg" --data-binary "@frame175751278672.jpg"```
"""

tags_metadata = [
    {
        "name":"Root",
        "description":"Simply what is behind the front page!"
    },
    {
        "name":"Test",
        "description":"In case of an error, double-check what you send - maybe it's in the wrong format! Request example below."
    },
    {
        "name":"Predict",
        "description":"Have the image you sent analyzed by WAKEE!"
    },
    {
        "name":"Backup",
        "description":"Alternative to predict above without personalized recommendations, should it fail. We run on free resources after all!"
    }
]

app = FastAPI(
    title="WAKEE.reloaded image analysis",
    description=description,
    version="0.veryconceptual",
    openapi_tags=tags_metadata
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", tags=["Root"], response_class=HTMLResponse)
async def read_root(request: Request):
    """Core website display."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/test", tags=["Test"], response_class=HTMLResponse)
@limiter.limit("5/minute")
async def check_request(request: Request):
    """Simple test endpoint returning the exact content of the request (the principle of the "dumb mirror"). Remember to:
    * `cd` into the right folder first,
    * provide an image (our intended input - surprising, I know!),
    * adapt the header depending your file format (like `-H "Content-Type: image/png"` if you send a png file),
    * make sure to be sending binary data (use `--data-binary`, **not** `-d`) **AND** keep the `@` before the filename,
    * and finally specify an `--output`!
    For some examples:
    
    **Linux/Mac**
    ```curl -X POST https://mevelios-wakee-reloaded-api.hf.space/test -H "Content-Type: image/jpeg" --data-binary @frame175751278672.jpg --output wakee_testing.jpg```
    
    **Windows**
    ```curl.exe -X POST https://mevelios-wakee-reloaded-api.hf.space/test -H "Content-Type: image/jpeg" --data-binary "@frame175751278672.jpg" --output "wakee_testing.jpg"```"""
    body = await request.body()
    return Response(
        content=body,
        media_type=request.headers.get("content-type", "application/octet-stream")
    )


@app.post("/predict", tags=["Predict"], response_class=HTMLResponse)
@limiter.limit("2/minute")
async def analyze_drift(request: Request):
    """Function communicating both ways with our CNN, matching WAKEE's core function. From the received image:
    * Captured frame (recovered from the request) is sent as CNN input,\n
    * CNN returns float values as output representing how strong emotions were recognized,\n
    * Said values are then matched to their labels so we may extract the most representative one,\n
    * Before sollicitating our LLM to generate a supportive message if markers of cognitive drift are identified.\n
    Given the focus on ADHD, very low engagement (or "disengagement") is given priority in recognition.\n
    Amongst the other three, only the top representative is kept. All other cases mean no drifting markers were identified!"""
    body = await request.body()
    try:
        pilimage = Image.open(BytesIO(body)).convert("RGB")
        cnn_predict = (cnn.get_emotion(pilimage))[0].tolist()
        dict_cnn = {"boredom" : cnn_predict[0], "confusion" : cnn_predict[1], "engagement" : cnn_predict[2], "frustration" : cnn_predict[3]}
        cnn_engagement = dict_cnn["engagement"]
        cnn_boredom = dict_cnn["boredom"]
        cnn_confusion = dict_cnn["confusion"]
        cnn_frustration = dict_cnn["frustration"]

        if cnn_engagement < 2.5:
            return llm.get_recommendation("disengagement")
        elif cnn_frustration > 0.5:
            return llm.get_recommendation("frustration")
        elif cnn_confusion > 0.61:
            return llm.get_recommendation("confusion")
        elif cnn_boredom > 1.05:
            return llm.get_recommendation("boredom")
        else:
            return "Good news, no cognitive drift recognized!"
    except Exception as exc:
        return f"Error in the process! Please use /backup endpoint for now. Displaying error message:\n{exc}"

@app.post("/backup", tags=["Backup"], response_class=HTMLResponse)
@limiter.limit("2/minute")
async def backup_analysis(request: Request):
    """Similar function to the /predict endpoint, minus the recommendation generator - instead, you will receive the identified emotion and a short, unvarying message."""
    body = await request.body()
    try:
        pilimage = Image.open(BytesIO(body)).convert("RGB")
        cnn_predict = (cnn.get_emotion(pilimage))[0].tolist()
        dict_cnn = {"boredom" : cnn_predict[0], "confusion" : cnn_predict[1], "engagement" : cnn_predict[2], "frustration" : cnn_predict[3]}
        cnn_engagement = dict_cnn["engagement"]
        cnn_boredom = dict_cnn["boredom"]
        cnn_confusion = dict_cnn["confusion"]
        cnn_frustration = dict_cnn["frustration"]

        if cnn_engagement < 2.5:
            return "Disengagement: careful, you're losing focus!"
        elif cnn_frustration > 0.5:
            return "Frustration: maybe it's time for a pause?"
        elif cnn_confusion > 0.61:
            return "Confusion: ask someone else's opinion on what you do not understand?"
        elif cnn_boredom > 1.05:
            return "Boredom: a short walk to get back into things!"
        else:
            return "Good news: no cognitive drift recognized!"
    except Exception as exc:
        return exc

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860) 