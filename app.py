import io
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# সর্বোচ্চ ৩০ মেগাবাইট পর্যন্ত ইমেজ আপলোড সাপোর্ট করবে
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024


def fast_thinning(img):
    """
    Morphological Thinning/Skeletonization Algorithm.
    মোটা দাগের ঠিক মাঝখানের ১-পিক্সেল সিঙ্গেল লাইন (Centerline) বের করার ইঞ্জিন।
    """
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            break
    return skel


def trace_centerline_to_svg(binary_skeleton, width, height, smooth_tolerance=1.5):
    """
    সিঙ্গেল পিক্সেল লাইনগুলোকে স্মুথ ভেক্টর পাথে কনভার্ট করে স্ট্যান্ডার্ড SVG তৈরি করে।
    """
    # 8-connectivity দিয়ে লাইনগুলোকে আলাদা ট্রেস করা
    contours, _ = cv2.findContours(
        binary_skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    svg_paths = []
    for cnt in contours:
        # অপ্রয়োজনীয় ছোটখাটো ডট বা নয়েজ বাতিল করা (কমপক্ষে ৫ পিক্সেল দীর্ঘ হতে হবে)
        if len(cnt) < 5:
            continue

        # Douglas-Peucker অ্যালগরিদম দিয়ে লাইন স্মুথ করা (হাত কাঁপা ঠিক করা)
        approx = cv2.approxPolyDP(cnt, smooth_tolerance, False)

        pts = approx.reshape(-1, 2)
        if len(pts) < 2:
            continue

        # SVG Path জেনারেট করা
        path_data = f"M {pts[0][0]} {pts[0][1]}"
        for pt in pts[1:]:
            path_data += f" L {pt[0]} {pt[1]}"

        # Adobe Stock স্ট্যান্ডার্ড: fill="none" এবং stroke="#000000"
        svg_paths.append(
            f'<path d="{path_data}" fill="none" stroke="#000000" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />'
        )

    # পূর্ণাঙ্গ ভেক্টর SVG স্ট্রাকচার
    paths_str = "\n  ".join(svg_paths)
    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">
  <g id="Editable_Stroke_Icons">
  {paths_str}
  </g>
</svg>"""
    return svg_content, len(svg_paths)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/trace", methods=["POST"])
def trace_image():
    if "image" not in request.files:
        return jsonify({"error": "কোনো ইমেজ পাওয়া যায়নি!"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "ফাইল সিলেক্ট করা হয়নি!"}), 400

    # মেমোরিতে ফাইল পড়া (Zero Disk Write)
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    if img is None:
        return jsonify({"error": "ইমেজ ফরম্যাট সাপোর্ট করছে না!"}), 400

    # আলফা চ্যানেল (ট্রান্সপারেন্ট ব্যাকগ্রাউন্ড) থাকলে সাদা দিয়ে রিপ্লেস করা
    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha_channel = img[:, :, 3]
        rgb_channels = img[:, :, :3]
        white_bg = np.ones_like(rgb_channels, dtype=np.uint8) * 255
        alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
        img = (rgb_channels * alpha_factor + white_bg * (1 - alpha_factor)).astype(np.uint8)

    # গ্রেস্কেল রূপান্তর
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    height, width = gray.shape

    # থ্রেশহোল্ড ভ্যালু ফ্রন্টএন্ড থেকে পাওয়া (ডিফল্ট অটো-ওটসু)
    threshold_val = request.form.get("threshold", type=int, default=200)

    # ব্যাকগ্রাউন্ড রিমুভ ও বাইনারি মাস্ক তৈরি (দাগ সাদা, ব্যাকগ্রাউন্ড কালো)
    _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)

    # সেন্টারলাইন (সিঙ্গেল পাথ) এক্সট্র্যাক্ট করা
    skeleton = fast_thinning(binary)

    # স্মুথিং টলারেন্স (ডিফল্ট ১.২)
    smooth = request.form.get("smooth", type=float, default=1.2)

    # ভেক্টর SVG তৈরি
    svg_data, path_count = trace_centerline_to_svg(skeleton, width, height, smooth)

    return jsonify({
        "status": "success",
        "svg": svg_data,
        "paths_detected": path_count,
        "width": width,
        "height": height
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
