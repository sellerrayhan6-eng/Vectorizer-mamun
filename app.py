import io
import base64
import re
import requests
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024

# তোমার API Key (যদি তোমার দেওয়া কি কাজ না করে, তবে AIzaSy... কি এখানে বসাবে)
GEMINI_API_KEY = "AQ.Ab8RN6Kmr6A6Ig4oNvESl1r05fnhwguEH2mJ-ufkfjfn6nWsVg"


def clean_svg_output(ai_text):
    """
    AI-এর রেসপন্স থেকে খাঁটি SVG কোড আলাদা করার ফিল্টার।
    """
    # কোড ব্লক থাকলে ক্লিন করা
    match = re.search(r'<svg[\s\S]*?<\/svg>', ai_text)
    if match:
        svg_code = match.group(0)
    else:
        svg_code = ai_text.strip()

    # Adobe Stock রুলস নিশ্চিত করা (fill="none" এবং stroke="#000000")
    if 'fill="none"' not in svg_code:
        svg_code = svg_code.replace('<svg', '<svg fill="none"')

    return svg_code


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/trace", methods=["POST"])
def trace_with_ai():
    if "image" not in request.files:
        return jsonify({"error": "কোনো ইমেজ পাওয়া যায়নি!"}), 400

    file = request.files["image"]
    image_bytes = file.read()
    
    # ইমেজ সাইজ বের করা
    img = Image.open(io.BytesIO(image_bytes))
    width, height = img.size

    # বেস৬৪ এনকোডিং
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    # এআইকে প্রফেশনাল ভেক্টর ডিজাইনার হিসেবে নির্দেশ দেওয়া
    prompt = f"""
    You are a world-class vector icon designer for Adobe Stock.
    Look at this icon plate very carefully.
    
    TASK:
    Recreate EVERY single icon from this sheet into high-quality, professional, single-path SVG vectors.
    
    CRITICAL ADOBE STOCK REQUIREMENTS:
    1. STRICT SINGLE STROKE PATHS: Do NOT use outlines or double-lines. Every icon line must be an open centerline path or pure circle/arc.
    2. Fill MUST be "none" everywhere: fill="none".
    3. Stroke MUST be black: stroke="#000000".
    4. Stroke-width="2", stroke-linecap="round", stroke-linejoin="round".
    5. Maintain the EXACT design, meaning, and layout of every icon from the image. Heads must be clean <circle>, bodies and lines must be ultra-smooth <path> curves.
    6. Return ONLY the raw valid SVG code starting with <svg viewBox="0 0 {width} {height}"> and ending with </svg>. No markdown, no explanations.
    """

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64_image
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 8192
        }
    }

    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
        res_data = response.json()

        if "error" in res_data:
            return jsonify({"error": f"API Error: {res_data['error'].get('message', 'Key সমস্যা')}"}), 400

        ai_svg = res_data["candidates"][0]["content"]["parts"][0]["text"]
        pure_svg = clean_svg_output(ai_svg)

        # পাথের সংখ্যা গোনা
        path_count = pure_svg.count("<path") + pure_svg.count("<circle") + pure_svg.count("<line")

        return jsonify({
            "status": "success",
            "svg": pure_svg,
            "paths_detected": path_count,
            "width": width,
            "height": height
        })

    except Exception as e:
        return jsonify({"error": f"প্রসেসিং ফেইল্ড: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
