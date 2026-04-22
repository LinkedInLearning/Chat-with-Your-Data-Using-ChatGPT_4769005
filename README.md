# Chat with Your Data Using ChatGPT
This is the repository for the LinkedIn Learning course `Chat with Your Data Using ChatGPT`. The full course is available from [LinkedIn Learning][lil-course-url].

![course-name-alt-text][lil-thumbnail-url] 

## Requirements
- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/account/api-keys)

## Setup

1. **Clone this repo** (or download the files).
2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate   # macOS/Linux
    venv\Scripts\activate      # Windows
    ```
3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4. **Set your OpenAI API key or place in .env file**:
    ```bash
    export OPENAI_API_KEY="your_api_key"      # macOS/Linux
    setx OPENAI_API_KEY "your_api_key"        # Windows PowerShell
    ```

## Running the Examples

Run the main demo script to see all lessons in action:

```bash
python research_agent.py
```


[lil-thumbnail-url]: https://media.licdn.com/dms/image/v2/D560DAQHCcPWvis_ffA/learning-public-crop_675_1200/B56ZxyEr7OHsAY-/0/1771440357483?e=2147483647&v=beta&t=Z7diLKsoASBRRmoM8iZB22IAH-NS6vRTSIVkxxDk9NM
