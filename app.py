import io
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024


def heal_and_skeletonize(binary):
    """
    ভাঙা লাইনগুলোকে জোড়া লাগানো (Healing) এবং 
    নিখুঁত ১-পিক্সেল সিঙ্গেল লাইন (Skeleton) বের করার অ্যালগরিদম।
    """
    # ১. লাইনগুলোকে আগে শক্ত ও জোড়া লাগানো (Closing)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    healed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # ২. নিখুঁত সেন্টারলাইন বের করা (Guo-Hall / Morphological Thinning)
    skel = np.zeros(healed.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    temp = np.empty(healed.shape, np.uint8)
    eroded = np.empty(healed.shape, np.uint8)

    while True:
        cv2.erode(healed, element, eroded)
        cv2.dilate(eroded, element, temp)
        cv2.subtract(healed, temp, temp)
        cv2.bitwise_or(skel, temp, skel)
        healed, eroded = eroded, healed

        if cv2.countNonZero(healed) == 0:
            break

    return skel


def trace_to_clean_svg(skel, orig_w, orig_h, scale_factor, smooth_tol=1.5):
    """
    সিঙ্গেল লাইনগুলোকে জোড়া দিয়ে লম্বা স্মুথ পাথে রূপান্তর করা।
    """
    # ভাঙা লাইন ফিল্টার করতে ছোট গাউসিয়ান ব্লার ও থ্রেশহোল্ড
    contours, _ = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

    svg_paths = []
    for cnt in contours:
        # ছোটখাটো আজেবাজে ডট বা নয়েজ বাতিল (কমপক্ষে ১৫ পিক্সেল বড় হতে হবে)
        if cv2.arcLength(cnt, False) < 15:
            continue

        # লাইন সোজা ও স্মুথ করা
        approx = cv2.approxPolyDP(cnt, smooth_tol, False)
        pts = approx.reshape(-1, 2)

        if len(pts) < 2:
            continue

        # বড় করা স্কেলকে আবার অরিজিনাল সাইজে ফিরিয়ে আনা
        orig_pts = pts / scale_factor

        path_data = f"M {orig_pts[0][0]:.1f} {orig_pts[0][1]:.1f}"
        for pt in orig_pts[1:]:
            path_data += f" L {pt[0]:.1f} {pt[1]:.1f}"

        svg_paths.append(
            f'<path d="{path_data}" fill="none" stroke="#000000" '
            f'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />'
        )

    paths_str = "\n  ".join(svg_paths)
    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {orig_w} {orig_h}" width="100%" height="100%">
  <g id="Editable_Stroke_Icons">
  {paths_str}
  </g>
</svg>"""
    return svg_content, len(svg_paths)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/trace", methods=["POST"])
def trace():
    if "image" not in request.files:
        return jsonify({"error": "কোনো ইমেজ পাওয়া যায়নি!"}), 400

    file = request.files["image"]
    in_memory = io.BytesIO()
    file.save(in_memory)
    data = np.frombuffer(in_memory.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    if img is None:
        return jsonify({"error": "ভুল ফরম্যাট!"}), 400

    # আলফা চ্যানেল ফিক্স
    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        rgb = img[:, :, :3]
        white = np.ones_like(rgb, dtype=np.uint8) * 255
        af = alpha[:, :, np.newaxis] / 255.0
        img = (rgb * af + white * (1 - af)).astype(np.uint8)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    orig_h, orig_w = gray.shape

    # ১. সুপার রেজোলিউশন স্কেলিং (লাইন যাতে না ফাটে)
    target_width = 2000
    scale = max(1.0, target_width / float(orig_w))
    if scale > 1.0:
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        processed = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        scale = 1.0
        processed = gray

    # ২. ব্যাকগ্রাউন্ড থেকে কালো লাইন ফিল্টার করা (Adaptive Otsu)
    thresh_user = request.form.get("threshold", type=int, default=200)
    # হালকা ব্লার দিয়ে ছবির নয়েজ ও কম্প্রেশন দূর করা
    blurred = cv2.GaussianBlur(processed, (3, 3), 0)
    _, binary = cv2.threshold(blurred, thresh_user, 255, cv2.THRESH_BINARY_INV)

    # ৩. লাইন রিপেয়ার ও সিঙ্গেল পাথ ট্রেসিং
    skel = heal_and_skeletonize(binary)

    smooth_user = request.form.get("smooth", type=float, default=1.5)
    svg_data, count = trace_to_clean_svg(skel, orig_w, orig_h, scale, smooth_user)

    return jsonify({
        "status": "success",
        "svg": svg_data,
        "paths_detected": count,
        "width": orig_w,
        "height": orig_h
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
