🧠 Image-Generation-with-Pre-trained-Models


Welcome to Task 2 of the Generative AI Internship at Prodigy InfoTech! This project is a Flask-based Image Generation Web App using the Stable Diffusion model via Hugging Face Diffusers. It provides a user-friendly interface to generate AI-powered images based on prompts.

🌟 Features

🎨 Real-time image generation using Stable Diffusion models (Realisian, Waifu, OrangeMix)

🧠 Prompt-based inference with validation and async generation

🧾 History tracking and deletion (with timestamps)

📊 Live usage statistics per model

🔥 Auto-cleanup of old files

🌐 Flask-based web interface with AJAX and JSON responses

🛠 Tech Stack

Layer

Technology

Backend

Python, Flask, Diffusers (Hugging Face)

Frontend

HTML, CSS, JS (Vanilla, AJAX), Jinja2

Models

runwayml/stable-diffusion-v1-5, waifu-diffusion, OrangeMix fallback

Deployment

Runs locally via Flask / can be hosted on Colab

✅ Getting Started Locally

1. Clone the Repository

git clone https://github.com/your-username/generative-ai-internship-t2.git
cd generative-ai-internship-t2

2. Set Up Environment

python -m venv venv
venv\Scripts\activate      # On Windows
# OR
source venv/bin/activate   # On macOS/Linux

pip install -r requirements.txt

3. Run the Flask App

python app.py

Visit http://127.0.0.1:5000/ in your browser to interact with the app.

📁 Folder Structure

.
├── app.py                        # Main Flask server
├── templates/
│   └── index.html                # Frontend UI
├── static/
│   ├── images/                   # Generated images saved here
│   ├── styles.css / scripts.js   # Optional assets
├── history.json                  # Stores image history
├── stats.json                    # Stores usage stats
├── requirements.txt
├── README.md

🔐 Note on Large Files

Some files such as generated images and model weights are not pushed due to size limits.
Please download your own models if you're customizing the backend pipeline further.

🧪 Google Colab Version

You can try this app online via Colab:

👉 🔗 Open in Google Colab

👨‍💻 Author

Mishal K S🎓 MCA Student, Jain University💼 Generative AI Intern @ Prodigy InfoTech

📜 License

This project is provided for academic and demonstration purposes only. All rights reserved.

Feel free to ⭐ star the repo if you find it useful!
