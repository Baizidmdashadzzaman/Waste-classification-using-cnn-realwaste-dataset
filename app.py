from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
import numpy as np

app = Flask(__name__)

# ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation']
class_names = ['কার্ডবোর্ড', 'খাবার ও জৈব বর্জ্য', 'কাঁচ', 'ধাতব', 'অন্যান্য বর্জ্য', 'কাগজ', 'প্লাস্টিক',
               'টেক্সটাইল বর্জ্য', 'উদ্ভিজ্জ']
num_classes = len(class_names)

model_ft = models.vgg16(weights=None)
num_ftrs = model_ft.classifier[6].in_features
model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

try:
    model_ft.load_state_dict(torch.load('best_waste_classifier_vgg16.pth', map_location=torch.device('cpu')))
    model_ft.eval()
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model not found.")
    model_ft = None

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_image(image_bytes):
    """
    Predicts the class of an image and returns a list of class probabilities.
    """
    if model_ft is None:
        return [{'class_name': name, 'probability': 0.0} for name in class_names]

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = data_transforms(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model_ft(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        probabilities_list = probabilities.numpy()

    prediction_results = []
    for i, prob in enumerate(probabilities_list):
        prediction_results.append({
            'class_name': class_names[i],
            'probability': float(prob * 100)
        })

    prediction_results.sort(key=lambda x: x['probability'], reverse=True)
    return prediction_results


@app.route('/')
def index():
    """
    Main page for the application.
    """
    return render_template('index.html')


@app.route('/info')
def info_page():
    """
    Information page about the waste types.
    """
    info = {
        'কার্ডবোর্ড': 'কার্ডবোর্ড পুনর্ব্যবহারযোগ্য একটি উপাদান, যা মূলত প্যাকেজিংয়ে ব্যবহৃত হয়।',
        'খাবার ও জৈব বর্জ্য': 'খাবার ও জৈব বর্জ্য কম্পোস্ট বা বায়োগ্যাস উৎপাদনে ব্যবহৃত হতে পারে।',
        'কাঁচ': 'কাঁচ পুনর্ব্যবহারযোগ্য এবং শতভাগ রিসাইকেল করা যায়।',
        'ধাতব': 'ধাতব বর্জ্য, যেমন টিন ও অ্যালুমিনিয়াম, গলিয়ে নতুন পণ্য তৈরি করা যায়।',
        'অন্যান্য বর্জ্য': 'এই শ্রেণীতে এমন বর্জ্য অন্তর্ভুক্ত, যা উপরের কোন শ্রেণীতে পড়ে না এবং সাধারণত ল্যান্ডফিলে যায়।',
        'কাগজ': 'কাগজ সহজেই পুনর্ব্যবহারযোগ্য এবং নতুন কাগজ পণ্য তৈরি করতে ব্যবহৃত হয়।',
        'প্লাস্টিক': 'প্লাস্টিক পুনর্ব্যবহারযোগ্য হলেও, এর প্রকারভেদ অনুযায়ী প্রক্রিয়া ভিন্ন হয়।',
        'টেক্সটাইল বর্জ্য': 'পুরাতন কাপড় বা অন্যান্য টেক্সটাইল পণ্য যা পুনর্ব্যবহার বা পুনরায় ব্যবহার করা যেতে পারে।',
        'উদ্ভিজ্জ': 'গাছপালা, পাতা এবং অন্যান্য প্রাকৃতিক উপাদান, যা কম্পোস্ট করা যায়।'
    }
    return render_template('info.html', info=info)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles image uploads and returns prediction results.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if file:
        try:
            img_bytes = file.read()
            predictions = predict_image(img_bytes)
            return jsonify({'predictions': predictions})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Something went wrong.'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
