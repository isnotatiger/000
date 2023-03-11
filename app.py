import gradio as gr

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")

preset_examples = [
    ('Instruction: given a dialog context, you need to response empathically.',
     '', 'Who are you?', 'Chitchat'),
]


def generate(instruction, knowledge, dialog, top_p, min_length, max_length):
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"

    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids
    outputs = model.generate(input_ids, min_length=int(
        min_length), max_length=int(max_length), top_p=top_p, do_sample=True)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(query)
    print(output)
    return output


def api_call_generation(instruction, knowledge, query, top_p, min_length, max_length):

    dialog = [
        query
    ]
    response = generate(instruction, knowledge, dialog,
                        top_p, min_length, max_length)

    return response


def change_example(choice):
    choice_idx = int(choice.split()[-1]) - 1
    instruction, knowledge, query, instruction_type = preset_examples[choice_idx]
    return [gr.update(lines=1, visible=True, value=instruction), gr.update(visible=True, value=knowledge), gr.update(lines=1, visible=True, value=query), gr.update(visible=True, value=instruction_type)]

def change_textbox(choice):
    if choice == "Chitchat":
        return gr.update(lines=1, visible=True, value="Instruction: given a dialog context, you need to response empathically.")
    elif choice == "Grounded Response Generation":
        return gr.update(lines=1, visible=True, value="Instruction: given a dialog context and related knowledge, you need to response safely based on the knowledge.")
    else:
        return gr.update(lines=1, visible=True, value="Instruction: given a dialog context and related knowledge, you need to answer the question based on the knowledge.")


with gr.Blocks() as demo:
    gr.Markdown("# This is a chat bot based off of Microsoft GODEL")
    gr.Markdown('''Ask me anything''')

    dropdown = gr.Dropdown(
        [f"Example {i+1}" for i in range(1)], label='Examples')

    radio = gr.Radio(
        ["Conversational Question Answering", "Chitchat", "Grounded Response Generation"], label="Instruction Type", value='Conversational Question Answering'
    )
    instruction = gr.Textbox(lines=1, interactive=True, label="Instruction",
                             value="Instruction: given a dialog context and related knowledge, you need to answer the question based on the knowledge.")
    radio.change(fn=change_textbox, inputs=radio, outputs=instruction)
    knowledge = gr.Textbox(lines=6, label="Knowledge")
    query = gr.Textbox(lines=1, label="User Query")

    dropdown.change(change_example, dropdown, [instruction, knowledge, query, radio])

    with gr.Row():
        with gr.Column(scale=1):
            response = gr.Textbox(label="Response", lines=2)

        with gr.Column(scale=1):
            top_p = gr.Slider(0, 1, value=0.9, label='top_p')
            min_length = gr.Number(8, label='min_length')
            max_length = gr.Number(
                64, label='max_length (should be larger than min_length)')

    greet_btn = gr.Button("Generate")
    greet_btn.click(fn=api_call_generation, inputs=[
                    instruction, knowledge, query, top_p, min_length, max_length], outputs=response)

demo.launch()