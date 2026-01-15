import gradio as gr
import os
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

# Global variable to store chat history for the current session
current_chat_history = []

# Define the function for chatbot response
def respond(
    message,
    history,  # gr.ChatInterface uses list of dicts format
    system_message,
    max_tokens,
    temperature,
    top_p,
    hf_token: gr.OAuthToken,
):
    global current_chat_history
    
    # Build messages list for the API
    messages = [{"role": "system", "content": system_message}]
    
    # Add conversation history (gr.ChatInterface uses list of dicts format)
    for msg in history:
        messages.append(msg)
        # Also add to our text history for download
        if msg["role"] == "user":
            current_chat_history.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            current_chat_history.append(f"Assistant: {msg['content']}")
    
    # Add current message
    messages.append({"role": "user", "content": message})
    current_chat_history.append(f"User: {message}")

    client = InferenceClient(token=hf_token.token, model="openai/gpt-oss-20b")

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        choices = message.choices
        token = ""
        if len(choices) and choices[0].delta.content:
            token = choices[0].delta.content

        response += token
        yield response

    # Append the assistant's final response to the history
    current_chat_history.append(f"Assistant: {response}")

def download_chat_history():
    # Join the current chat history into a single string
    history_str = "\n".join(current_chat_history)
    # Save the chat history to a text file
    with open("chat_history.txt", "w") as f:
        f.write(history_str)
    return "chat_history.txt"

def clear_chat_history():
    # Reset the current chat history
    global current_chat_history
    current_chat_history.clear()  # Clear the chat history
    return "Chat history cleared."

# Excel reading function
def read_excel(file):
    df = pd.read_excel(file.name)
    return df.to_string()

# Create the main chatbot interface using gr.ChatInterface for the Detailed Analysis tab
def create_career_chatbot():
    return gr.ChatInterface(
        respond,
        type="messages",
        additional_inputs=[
            gr.Textbox(value=system_message, label="System message", visible=False),
            gr.Slider(minimum=1, maximum=2048, value=1024, step=1, label="Max new tokens", visible=False),
            gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature", visible=False),
            gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)", visible=False),
        ],
    )

# Create the simple chatbot interface for General Guidance tab
def create_simple_chatbot():
    return gr.ChatInterface(
        respond,
        type="messages",
        additional_inputs=[
            gr.Textbox(value="You are an AI powered chatbot named as Career Compass built by Hashir Ehtisham who is a Computer Engineering student of NUST CEME to help students, teachers, and parents find the best career paths based on students' interests and academic performance.", 
                      label="System message", visible=False),
            gr.Slider(minimum=1, maximum=2048, value=1024, step=1, label="Max new tokens", visible=False),
            gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature", visible=False),
            gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)", visible=False),
        ],
    )

# Create the Gradio interface
with gr.Blocks(css=css) as demo:
    # Add login button to sidebar
    with gr.Sidebar():
        gr.LoginButton()
        download_button = gr.Button("Download Chat History")
        clear_button = gr.Button("Clear Chat History")
        status_output = gr.Textbox(label="Status", interactive=False)
        download_output = gr.File(label="Download")
        
        download_button.click(download_chat_history, outputs=download_output)
        clear_button.click(clear_chat_history, outputs=status_output)
    
    # Introduction Tab
    with gr.Tab("Career Compass"):
        with gr.Row(elem_id="image-container"):
            gr.Image(image_url, elem_id="compass-image")
        
        gr.Markdown("# **Career Compass**")
        gr.Markdown("### **Developed by Hashir Ehtisham**")
        gr.Markdown("""
        **Career Compass** is a cutting-edge AI-powered tool designed to provide personalized career guidance based on students' academic performance and extracurricular activities. The key features of this tool include:
        - **Personalized Analysis:** Delivers career advice tailored to individual student profiles.
        - **Streamlined Interface:** Simple and intuitive user experience.
        - **Detailed Reports:** Offers insights into suitable career paths, relevant universities, and job opportunities.
        - **General Guidance & Emotional Support:** Talk to AI for General Career Guidance and also lighten your mood. 
        
        **How It Works:**
        - **Detailed Analysis**
        1. Upload your academic records.
        2. Input your query regarding career guidance.
        3. Get detailed recommendations and potential career paths.
        4. Download the Report!
        - **General Guidance & Emotional Support**
        1. Enter your query and doubts about choosing University majors.
        2. Ask about Career Opportunities and scope of different fields.
        3. Get unbiased AI analyzed answers and recommendations!
        """)
    
    # Detailed Analysis Tab
    with gr.Tab("Detailed Analysis"):
        gr.Markdown("# Detailed Analysis")
        gr.Markdown("Get personalized career guidance based on academic performance and extracurricular activities.\n<div style='color: green;'>Developed by Hashir Ehtisham</div>")
        career_chatbot = create_career_chatbot()
    
    # File Upload Tab
    with gr.Tab("Upload Data"):
        gr.Markdown("# Upload Data")
        gr.Markdown("Upload your academic record along with extracurricular activities here.\n<div style='color: green;'>Don't worry if your extracted data appears a bit strange. 😉 </div> \n<div style='color: green;'>Developed by Hashir Ehtisham</div>")
        file_input = gr.File(label="Upload Excel file")
        excel_output = gr.Textbox(label="Excel Content")
        file_input.change(read_excel, inputs=file_input, outputs=excel_output)

    # Simple Chatbot Tab
    with gr.Tab("General Guidance & Emotional Support"):
        gr.Markdown("# General Guidance & Emotional Support")
        gr.Markdown("A compassionate career counseling chatbot providing personalized guidance on career paths and emotional support. \n<div style='color: green;'>Developed by Hashir Ehtisham</div>")
        simple_chatbot = create_simple_chatbot()

if __name__ == "__main__":
    demo.launch()
