import os
import random
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps, ImageFilter, ImageEnhance
import cv2

# CONFIGURATION
FONT_PATH = r"C:\Users\hussa\OneDrive\Desktop\PROJECT\creating tamil characters\Kavivanar-Regular.ttf"
OUTPUT_DIR = "tamil_dataset_full_augmented"
IMAGE_SIZE = 128
NUM_IMAGES = 1000  # Images per character

# All 247 Tamil characters (Uyir, Mei, Uyirmei, Grantha, etc.)
TAMIL_CHARACTERS = [
    # Uyir
    "அ", "ஆ", "இ", "ஈ", "உ", "ஊ", "எ", "ஏ", "ஐ", "ஒ", "ஓ", "ஔ",
    
    # Mei
    "க்", "ங்", "ச்", "ஞ்", "ட்", "ண்", "த்", "ந்", "ப்", "ம்", "ய்", "ர்", "ல்", "வ்", "ழ்", "ள்", "ற்", "ன்",

    # Uyirmei (consonant + vowel combinations)
    "க", "கா", "கி", "கீ", "கெ", "கே", "கை", "கொ", "கோ", "கௌ",
    "ங", "ஙா", "ஙி", "ஙீ", "ஙெ", "ஙே", "ஙை", "ஙொ", "ஙோ", "ஙௌ",
    "ச", "சா", "சி", "சீ", "செ", "சே", "சை", "சொ", "சோ", "சௌ",
    "ஞ", "ஞா", "ஞி", "ஞீ", "ஞெ", "ஞே", "ஞை", "ஞொ", "ஞோ", "ஞௌ",
    "ட", "டா", "டி", "டீ", "டெ", "டே", "டை", "டொ", "டோ", "டௌ",
    "ண", "ணா", "ணி", "ணீ", "ணெ", "ணே", "ணை", "ணொ", "ணோ", "ணௌ",
    "த", "தா", "தி", "தீ", "தெ", "தே", "தை", "தொ", "தோ", "தௌ",
    "ந", "நா", "நி", "நீ", "நெ", "நே", "நை", "நொ", "நோ", "நௌ",
    "ப", "பா", "பி", "பீ","பெ", "பே", "பை", "பொ", "போ", "பௌ",
    "ம", "மா", "மி", "மீ", "மெ", "மே", "மை", "மொ", "மோ", "மௌ",
    "ய", "யா", "யி", "யீ", "யெ", "யே", "யை", "யொ", "யோ", "யௌ",
    "ர", "ரா", "ரி", "ரீ","ரெ", "ரே", "ரை", "ரொ", "ரோ", "ரௌ",
    "ல", "லா", "லி", "லீ", "லெ", "லே", "லை", "லொ", "லோ", "லௌ",
    "வ", "வா", "வி", "வீ", "வெ", "வே", "வை", "வொ", "வோ", "வௌ",
    "ழ", "ழா", "ழி", "ழீ", "ழெ", "ழே", "ழை", "ழொ", "ழோ", "ழௌ",
    "ள", "ளா", "ளி", "ளீ","ளெ", "ளே", "ளை", "ளொ", "ளோ", "ளௌ",
    "ற", "றா", "றி", "றீ", "றெ", "றே", "றை", "றொ", "றோ", "றௌ",
    "ன", "னா", "னி", "னீ", "னெ", "னே", "னை", "னொ", "னோ", "னௌ",
    
    # Special/Grantha
    "ஃ", "ஜ", "ஷ", "ஸ", "ஹ", "க்ஷ"
]

def get_random_font_size(): return random.randint(50, 90)

def add_noise(img):
    np_img = np.array(img)
    noise = np.random.normal(0, 12, np_img.shape).astype(np.int16)
    noisy_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def elastic_transform(img, alpha=36, sigma=6):
    np_img = np.array(img)
    shape = np_img.shape
    dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (17, 17), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (17, 17), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    distorted = cv2.remap(np_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(distorted)

def apply_random_transformations(img):
    angle = random.uniform(-30, 30)
    img = img.rotate(angle, expand=True, fillcolor=255)
    scale_x = random.uniform(0.75, 1.25)
    scale_y = random.uniform(0.75, 1.25)
    img = img.resize((int(img.width * scale_x), int(img.height * scale_y)))
    img = ImageOps.pad(img, (IMAGE_SIZE, IMAGE_SIZE), color=255)
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    if random.random() < 0.3:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.2:
        img = elastic_transform(img)
    if random.random() < 0.3:
        img = add_noise(img)
    return img

def generate_dataset():
    for letter in TAMIL_CHARACTERS:
        folder_path = os.path.join(OUTPUT_DIR, letter)
        os.makedirs(folder_path, exist_ok=True)

        for i in range(NUM_IMAGES):
            font_size = get_random_font_size()
            try:
                font = ImageFont.truetype(FONT_PATH, size=font_size)
            except Exception as e:
                print(f"❌ Font load failed: {e}")
                continue

            canvas_size = 150
            img = Image.new("L", (canvas_size, canvas_size), color=255)
            draw = ImageDraw.Draw(img)

            bbox = font.getbbox(letter)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = random.randint(0, max(0, canvas_size - w))
            y = random.randint(0, max(0, canvas_size - h))
            draw.text((x, y), letter, font=font, fill=0)

            img = apply_random_transformations(img)

            try:
                img.save(os.path.join(folder_path, f"{letter}_{i+1}.png"))
            except Exception as e:
                print(f"❌ Error saving {letter}_{i+1}: {e}")

    print("✅ Tamil character dataset generation complete.")

if __name__ == "__main__":
    generate_dataset()
