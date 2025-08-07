# hoosmand_sam_api

import gradio as gr
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline

tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/gpt2-fa")
model = AutoModelWithLMHead.from_pretrained("HooshvareLab/gpt2-fa")
text_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

def reply_to_customer(message, tone):
    prompt = f"{tone} پاسخ به پیام: {message} "
    output = text_gen(prompt, max_length=100, num_return_sequences=1)[0]["generated_text"]
    return output.replace(prompt, "").strip()

iface = gr.Interface(
    fn=reply_to_customer,
    inputs=[
        gr.Textbox(label="پیام مشتری"),
        gr.Radio(["رسمی", "دوستانه"], label="لحن پاسخ", value="رسمی")
    ],
    outputs=gr.Textbox(label="پاسخ مدل"),
    title="هوشمند سام",
    description="👤 توسعه‌دهنده: امیر سام یعقوبی\n📧 amirsamyaghobi62@gmail.com\n📱 09366681963"
)

iface.launch()