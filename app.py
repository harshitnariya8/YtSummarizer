import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lyzr import ChatBot
import re

# Set your OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://youtubesummariser.netlify.app"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Dictionary of available prompts
prompts = {
    "summarize": """
        The summary should include:
        - The main topic or purpose of the video.
        - Key points or sections covered.
        - Any conclusions or important takeaways.
        
        Keep the summary clear and concise, ideally between 100-200 words.""",
    "actionable": """
    Act like a professional content strategist and summarization expert. 
    You have been helping individuals and organizations analyze, extract key insights, and present actionable steps from multimedia content for over 15 years.
    
    I have a YouTube video, and I want to identify the most valuable, actionable steps from it. My goal is to create a summary that highlights key takeaways users can apply immediately after watching. Please follow these steps:
    1. Watch or review the content carefully.
    2. Identify the main themes or concepts presented in the video.
    3. Break down the video content into key actionable steps. These steps should be specific actions that viewers can apply immediately to their own work, projects, or personal goals.
    4. Provide a step-by-step summary of each key action point, focusing on practical and immediately actionable steps.
    5. Format the response in a bullet point format with each point containing:
        - What the action is.
        - How to implement it.
        - Why this action is valuable or impactful.
    6. If any part of the content requires further clarification or background, include a brief context in the explanation for better understanding.
    7. Ensure the summary is detailed and comprehensive, so that even someone who hasn’t seen the video can benefit and apply the suggestions.
    Take a deep breath and work on this problem step-by-step.""",
    "quotes": """Act like a professional media analyst specializing in extracting insightful and impactful quotes from audio-visual content. You have over 15 years of experience analyzing speeches, interviews, and multimedia to identify key takeaways and essential quotes. You are now tasked with analyzing a YouTube video to generate key quotes that summarize the main points, impactful statements, and memorable moments.

    Instructions:
    Video Identification: Analyze the YouTube video provided. Ensure that the video is fully processed before extracting the quotes. If necessary, transcribe the audio using an accurate transcription tool. 
    Summarize Main Points: Begin by watching the video and identifying the main themes or arguments presented. Summarize the video content in a detailed paragraph to ensure you understand the context of the quotes you're extracting.   
    Quote Extraction: Extract at least 5-10 key quotes that:
    
    Highlight the main arguments or points being made.
    Reflect the most impactful or memorable moments in the video.
    Include statements that effectively capture the tone or emotions conveyed by the speaker.
    Ensure each quote is between 15 and 50 words in length and includes context, such as the time stamp and the speaker’s name if multiple speakers are present.
    
    Categorize Quotes: Categorize the quotes into relevant sections, such as:
    
    Main Arguments: Core messages or themes.
    Impactful Statements: Particularly emotional or memorable phrases.
    Supporting Facts: Quotes that provide evidence or data to back up arguments.
    Provide Citations: For each quote, include the video time stamp, and attribute it to the correct speaker. Ensure that the quotes accurately reflect the original meaning and tone.
    
    Formatting: Present the quotes in a clear format, with each quote on a new line, followed by the time stamp and speaker information in parentheses.
    
    Final Output: At the end of the prompt, summarize the overall significance of the quotes extracted and how they contribute to understanding the main message of the video.
    
    Take a deep breath and work on this problem step-by-step.""",
    "qa": """
    Act like an expert content summarizer and video analyst. You have been extracting key insights and information from video content for over 10 years. Your task is to watch a given YouTube video and create 10 relevant questions, each followed by a well-explained answer, based on the key points, themes, or content of the video.

    Instructions:
    Watch and Understand: Analyze the entire video carefully, identifying the main points, themes, and any specific information presented.
    Generate Questions: Create 10 questions that cover a broad range of the video’s content. Ensure the questions are well-structured, diverse, and designed to cover different aspects (such as key facts, deeper insights, or critical analysis).
    Provide Detailed Answers: After each question, give a detailed, informative, and accurate answer based on the video's content.
    Keep it Structured: For each question and answer pair, maintain this structure:
    Question: [Insert question here]
    Answer: [Insert detailed answer here]
    Content Breakdown: Ensure the answers reflect the video's core messages, providing both factual and interpretive responses when necessary.
    Example Structure:
    Question 1: What is the primary focus of the video, and how is it introduced by the speaker?
    
    Answer: The video primarily focuses on [insert topic], and it is introduced by [explanation based on the speaker’s introduction].
    Question 2: What are the main arguments or points made by the speaker throughout the video?
    
    Answer: The speaker presents [insert main arguments], elaborating on [details of the arguments].
    Note: Be thorough, ensuring that the questions and answers provide a comprehensive summary of the video’s content.
    
    Take a deep breath and work on this problem step-by-step.""",
}

# Define input model for the API request
class VideoRequest(BaseModel):
    url: str
    prompt_id: str


def extract_video_id(url):
    # Regular expression to match YouTube video IDs from different URL formats
    pattern = r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|embed/|v/|.+\?v=))([\w-]{11})"
    match = re.search(pattern, url)

    return match.group(1) if match else None

# FastAPI endpoint to analyze YouTube video
@app.post("/analyze-video")
async def analyze_video(video: VideoRequest):
    try:
        # Check if the provided prompt_id exists in the dictionary
        if video.prompt_id not in prompts:
            raise HTTPException(status_code=400, detail="Invalid prompt_id. Please use one of: 'summary', 'actionable_takeaways', 'weakness_in_reasoning'.")

        yt_url_id = extract_video_id(video.url)
        # Get the corresponding system prompt based on the prompt_id
        system_prompt = prompts[video.prompt_id]

        # Create a chatbot using the provided YouTube URL and system prompt
        chatbot = ChatBot.youtube_chat(urls=[yt_url_id], system_prompt=system_prompt)

        # Chat with the chatbot to generate a response
        response = chatbot.chat("")
        # response = {
        #   "response": "This video is a tutorial on how to use Q and2 Vision Language Model combined with Co Pali, a backend engine for running vision language models. The presenter demonstrates how to extract and ask questions from PDFs containing images, using a vision language model that understands images through an image encoder and orchestration tools for indexing documents and images. The tutorial is conducted on Google Colab Pro using a high-end GPU, and involves installing various libraries including Transformers, Accelerate, Flash Attention, Qen Utils, and PDF to Image, among others.\n\nThe presenter uses a case study PDF from Massachusetts General Hospital related to gastroesophageal surgery, which contains text and images. The goal is to ask questions related to the images in the document, showcasing the model's ability to understand and retrieve information from both text and visual data. The process involves converting the PDF into images, indexing the document using Co Pali (with BLD as a recommended wrapper for production use), and then performing a retrieval-augmented generation (RAG) process to generate answers to questions posed about the images.\n\nThe tutorial covers the installation of necessary libraries, setting up the environment, indexing the document, and writing code to perform the RAG process. The presenter demonstrates asking specific questions related to images in the document and shows how the model retrieves relevant information and generates responses. The video illustrates the power of combining vision and language models for document analysis, especially for documents containing multimodal data like text and images.\n\nThis tutorial is aimed at individuals interested in machine learning, natural language processing, and computer vision, particularly those looking to work with multimodal documents. It provides a practical example of how advanced AI models can be used to extract and interpret information from complex documents, offering insights into the capabilities and applications of vision language models."
        # }

        # Return the response from the chatbot
        return {"response": response.response}
        # return response

    except Exception as e:
        print("Error")
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

# To run the FastAPI server, use uvicorn:
# uvicorn main:app --reload