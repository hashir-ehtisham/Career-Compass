import gradio as gr
from huggingface_hub import InferenceClient
import pandas as pd

# Direct link to the image
image_url = "https://drive.google.com/uc?export=view&id=1AB7sFKxPLkJE_RmyUDap6fFaDlu1XGJl"

# Define the system message
system_message = """
You are a Career Counseling Chatbot. Analyze the student's academic performance and extracurricular activities to provide career guidance. Based on the provided data, respond in the following format and must include the following headings:
# **Student's Primary Interest with Reason**
# **Career Opportunities in the field**
# **Universities in Pakistan for related field**
# **Conclusion with name of field**
Ensure that the analysis is based on the student's performance in subjects and extracurriculars, and suggest relevant career options with details on possible high ranking universities in Pakistan.
"""

# CSS to hide footer, customize button, and center image
css = """
footer {display:none !important}
.output-markdown{display:none !important}
.gr-button-primary {
    z-index: 14;
    height: 43px;
    width: 130px;
    left: 0px;
    top: 0px;
    padding: 0px;
    cursor: pointer !important; 
    background: none rgb(17, 20, 45) !important;
    border: none !important;
    text-align: center !important;
    font-family: Poppins !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: rgb(255, 255, 255) !important;
    line-height: 1 !important;
    border-radius: 12px !important;
    transition: box-shadow 200ms ease 0s, background 200ms ease 0s !important;
    box-shadow: none !important;
}
.gr-button-primary:hover {
    z-index: 14;
    height: 43px;
    width: 130px;
    left: 0px;
    top: 0px;
    padding: 0px;
    cursor: pointer !important;
    background: none rgb(66, 133, 244) !important;
    border: none !important;
    text-align: center !important;
    font-family: Poppins !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: rgb(255, 255, 255) !important;
    line-height: 1 !important;
    border-radius: 12px !important;
    transition: box-shadow 200ms ease 0s, background 200ms ease 0s !important;
    box-shadow: rgb(0 0 0 / 23%) 0px 1px 7px 0px !important;
}
.hover\:bg-orange-50:hover {
    --tw-bg-opacity: 1 !important;
    background-color: rgb(229,225,255) !important;
}
.to-orange-200 {
    --tw-gradient-to: rgb(37 56 133 / 37%) !important;
}
.from-orange-400 {
    --tw-gradient-from: rgb(17, 20, 45) !important;
    --tw-gradient-to: rgb(255 150 51 / 0);
    --tw-gradient-stops: var(--tw-gradient-from), var(--tw-gradient-to) !important;
}
.group-hover\:from-orange-500 {
    --tw-gradient-from:rgb(17, 20, 45) !important; 
    --tw-gradient-to: rgb(37 56 133 / 37%);
    --tw-gradient-stops: var(--tw-gradient-from), var(--tw-gradient-to) !important;
}
.group:hover .group-hover\:text-orange-500 {
    --tw-text-opacity: 1 !important;
    color:rgb(37 56 133 / var(--tw-text-opacity)) !important;
}
#image-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: auto; /* Adjust the height as needed */
    margin-top: 20px; /* Adjust the margin as needed */
}
#compass-image {
    max-width: 800px; /* Adjust the width as needed */
    max-height: 600px; /* Adjust the height as needed */
    object-fit: contain; /* Maintains aspect ratio */
}
"""

# Initialize the InferenceClient for chatbot
client = InferenceClient("HuggingFaceH4/zephyr-7b-alpha")

# Define the function for chatbot response
def respond(
    message,
    history,
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

def send_message(message, history, system_message, max_tokens, temperature, top_p):
    if message:
        history.append((message, ""))
        response = respond(
            message=message,
            history=history,
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        response_text = ""
        for r in response:
            response_text = r
        # Apply HTML formatting to headings
        formatted_response_text = response_text.replace(
            "Student's Primary Interest with Reason:", "<h2><strong>Student's Primary Interest with Reason</strong></h2>"
        ).replace(
            "Career Opportunities in the field:", "<h2><strong>Career Opportunities in the field</strong></h2>"
        ).replace(
            "Universities in Pakistan for related field:", "<h2><strong>Universities in Pakistan for related field</strong></h2>"
        ).replace(
            "Conclusion with name of field:", "<h2><strong>Conclusion with name of field</strong></h2>"
        )
        history[-1] = (message, formatted_response_text)
    return history, gr.update(value="")

# Excel reading function
def read_excel(file):
    df = pd.read_excel(file.name)
    return df.to_string()

# Create the Gradio interface
with gr.Blocks(css=css) as demo:
    # Introduction Tab
    with gr.Tab("Career Compass"):
        with gr.Row(elem_id="image-container"):
            gr.Image(image_url, elem_id="compass-image")
        
        gr.Markdown("# **Career Compass**")
        gr.Markdown("### **Developed by Hashir (A student of APS DHA II Sec -D)**")
        gr.Markdown("""
        **Career Compass** is a cutting-edge AI-powered tool designed to provide personalized career guidance based on students' academic performance and extracurricular activities. The key features of this tool include:
        - **Personalized Analysis:** Delivers career advice tailored to individual student profiles.
        - **Streamlined Interface:** Simple and intuitive user experience.
        - **Detailed Reports:** Offers insights into suitable career paths, relevant universities, and job opportunities.
        - **Aptitude Test:** Take the Aptitude Test to determine your interest and find out the relevent field.
        **Libraries Used:**
        - **Gradio:** For creating the user interface.
        - **Pandas:** For reading and analyzing Excel files.
        - **Hugging API and LLM:** Zephyr-7b-beta For utilizing state-of-the-art language models.
        
        **How It Works:**
        - **Detailed Analysis**
        1. Upload your academic records.
        2. Input your query regarding career guidance.
        3. Get detailed recommendations and potential career paths!
         - **Aptitude test**
        1. Or choose to take "Aptitude test"
        2. Click on "generate an aptitude test for me (10 questions)"
        3. After that, 10 questions would appear.
        4. Answer those questions and submit the response.
        5. Get the AI analyzed answer and recommendations and potential career paths!
        """)
    
    # Detailed Analysis Tab
    with gr.Tab("Detailed Analysis"):
        gr.Markdown("# Detailed Analysis")
        gr.Markdown("Get personalized career guidance based on academic performance and extracurricular activities with Detailed Analysis.\n<div style='color: green;'>Developed by Hashir Ehtisham</div>")
        
        system_message_career = gr.Textbox(value=system_message, visible=False)
        chatbot_career = gr.Chatbot()
        msg_career = gr.Textbox(label="Enter the Excel Copied Data here")
        
        with gr.Row():
            clear_career = gr.Button("Clear")
            submit_career = gr.Button("Submit")
        
        with gr.Accordion("Additional Inputs", open=False):
            max_tokens_career = gr.Slider(minimum=1, maximum=2048, value=1024, step=1, label="Max new tokens")
            temperature_career = gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature")
            top_p_career = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)")

        def respond_wrapper_career(message, chat_history, system_message_val, max_tokens_val, temperature_val, top_p_val):
            chat_history, _ = send_message(
                message=message,
                history=chat_history,
                system_message=system_message_val,
                max_tokens=max_tokens_val,
                temperature=temperature_val,
                top_p=top_p_val,
            )
            return gr.update(value=chat_history), gr.update(value="")

        submit_career.click(
            respond_wrapper_career,
            inputs=[msg_career, chatbot_career, system_message_career, max_tokens_career, temperature_career, top_p_career],
            outputs=[chatbot_career, msg_career],
        )

        clear_career.click(lambda: None, None, chatbot_career, queue=False)

    # File Upload Tab
    with gr.Tab("Upload Data"):
        gr.Markdown("# Upload Data")
        file_input = gr.File(label="Upload Excel file")
        excel_output = gr.Textbox(label="Excel Content")
        file_input.change(read_excel, inputs=file_input, outputs=excel_output)
    
    # Aptitude Test Tab
    with gr.Tab("Aptitude Test"):
        gr.Markdown("# Aptitude Test")
        gr.Markdown("""
        Take the Aptitude Test to determine your interest and find out your interests towards different professional fields.
        <div style='color: gray;'>After generating the test, use the format e.g 1c,2b,3a,4c and more to answer the questions</div>
        <div style='color: green;'>Developed by Hashir Ehtisham</div>
        """)

        system_message_aptitude = gr.Textbox(value="You are an Aptitude Test Chatbot. Generate detailed mcq based questions and answers to determine that in what professional field does the the interest of student lies. Must include questions about Medical, engineering, computer science, arts and finance", visible=False)
        chatbot_aptitude = gr.Chatbot()
        msg_aptitude = gr.Textbox(label="Your message")

        with gr.Row():
            clear_aptitude = gr.Button("Clear")
            submit_aptitude = gr.Button("Submit")
            example_button = gr.Button("Generate Aptitude Test (10 questions)")

        with gr.Accordion("Additional Inputs", open=False):
            max_tokens_aptitude = gr.Slider(minimum=1, maximum=4096, value=2048, step=1, label="Max new tokens")
            temperature_aptitude = gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature")
            top_p_aptitude = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)")

        def respond_wrapper_aptitude(message, chat_history, system_message_val, max_tokens_val, temperature_val, top_p_val):
            chat_history, _ = send_message(
                message=message,
                history=chat_history,
                system_message=system_message_val,
                max_tokens=max_tokens_val,
                temperature=temperature_val,
                top_p=top_p_val,
            )
            return gr.update(value=""), chat_history

        def copy_to_message():
            return gr.update(value="Create a 10-question aptitude test designed to assess a student's interests in various academic fields such as Engineering, Medicine, Computer Science, Law, Finance and Arts. Each question should be multiple-choice with six options, and should ask the student about their preferences, interests, or inclinations towards activities, subjects, or scenarios related to these fields. Avoid questions that require prior knowledge in any specific subject. The test should be suitable for high school students exploring potential career paths.")
        submit_aptitude.click(
            respond_wrapper_aptitude,
            inputs=[
                msg_aptitude,
                chatbot_aptitude,
                system_message_aptitude,
                max_tokens_aptitude,
                temperature_aptitude,
                top_p_aptitude,
            ],
            outputs=[chatbot_aptitude],
        )
        msg_aptitude.submit(respond_wrapper_aptitude, [msg_aptitude, chatbot_aptitude, system_message_aptitude, max_tokens_aptitude, temperature_aptitude, top_p_aptitude], [msg_aptitude, chatbot_aptitude])

        clear_aptitude.click(lambda: None, None, chatbot_aptitude, queue=False)
        example_button.click(copy_to_message, [], [msg_aptitude])

    # Simple Chatbot Tab (new tab integration)
    with gr.Tab("General Guidance"):
        gr.Markdown("# General Guidance")
        gr.Markdown("""
        A compassionate career counseling chatbot providing personalized guidance on career paths and emotional support for your journey.
        <div style='color: green;'>Developed by Hashir Ehtisham</div>
        """)

        system_message_simple = gr.Textbox(value="You are an AI powered chatbot named as Career Compass built by Hashir Ehtisham who is a student of APS DHA II Sec -D to help students, teachers, and parents find the best career paths based on students' interests and academic performance.", visible=False)
        chatbot_simple = gr.Chatbot()
        msg_simple = gr.Textbox(label="Type a message")
        
        with gr.Row():
            clear_simple = gr.Button("Clear")
            submit_simple = gr.Button("Submit")
        
        with gr.Accordion("Additional Inputs", open=False):
            max_tokens_simple = gr.Slider(minimum=1, maximum=2048, value=1024, step=1, label="Max new tokens")
            temperature_simple = gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature")
            top_p_simple = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)")
        
        def respond_wrapper_simple(message, chat_history, system_message_val, max_tokens_val, temperature_val, top_p_val):
            chat_history, _ = send_message(
                message=message,
                history=chat_history,
                system_message=system_message_val,
                max_tokens=max_tokens_val,
                temperature=temperature_val,
                top_p=top_p_val,
            )
            return chat_history

        submit_simple.click(
            respond_wrapper_simple,
            inputs=[
                msg_simple,
                chatbot_simple,
                system_message_simple,
                max_tokens_simple,
                temperature_simple,
                top_p_simple,
            ],
            outputs=[chatbot_simple],
        )

        clear_simple.click(lambda: None, None, chatbot_simple)

# Launch the Gradio app
demo.launch()
