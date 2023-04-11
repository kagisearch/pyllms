import gradio as gr
from llms import LLMS


supported_models = [model["name"] for model in LLMS().list()]
default_model = supported_models[0]

with gr.Blocks() as demo:
    model_choice = gr.components.Dropdown(choices=supported_models,
                                          value=default_model,
                                          label="model_choice")
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history, model_choice):
        model = LLMS(model=model_choice)
        # TODO: support multi-turns messages
        last_message = history[-1][0]
        result = model.complete(last_message)
        bot_message = result.text
        history[-1][1] = bot_message
        return history

    msg.submit(fn=user,
               inputs=[msg, chatbot],
               outputs=[msg, chatbot],
               queue=False,
               ).then(
                fn=bot,
                inputs=[chatbot, model_choice],
                outputs=chatbot,
                )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
