import asyncio
import csv
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from assistant import lg_agent


app = FastAPI()

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Feedback(BaseModel):
    user_message: str
    bot_message: str
    feedback: str


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        async for message in websocket.iter_text():
            try:
                word = ""
                async for event in lg_agent.astream_events(
                    {"messages": ("user", message)},
                    config={"configurable": {"thread_id": 1}},
                    version="v1",
                ):
                    kind = event["event"]

                    if kind == "on_chat_model_stream":
                        token = event["data"]["chunk"].content
                        if token:
                            if token.startswith(" "):
                                if word:
                                    await websocket.send_text(word)
                                word = token
                            else:
                                word += token
                # Send the final partial message if any
                if word:
                    await websocket.send_text(word)
            except Exception as e:
                await websocket.send_text(f"Error: {e}")
    except Exception as e:
        await websocket.send_text(f"Error: {e}")


@app.post("/feedback")
async def receive_feedback(feedback: Feedback):
    with open("docs/feedback.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [feedback.user_message, feedback.bot_message, feedback.feedback]
        )
    return {"status": "success"}


# Serve the static HTML file
@app.get("/")
async def get():
    return HTMLResponse(open("static/index.html").read())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
