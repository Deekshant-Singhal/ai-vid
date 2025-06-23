from fastapi import APIRouter, Body
# from summarization.text.summarizer import summarize_text

router = APIRouter()

# # @router.post("/")
# # async def summarize(data: dict = Body(...)):
# #     text = data.get("text", "")
# #     mood = data.get("mood", "default")
# #     summary = summarize_text(text, mood=mood)
# #     return {"summary": summary}
