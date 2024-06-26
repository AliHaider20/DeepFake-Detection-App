"""# Emsembling Transformer and CNN"""

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import gradio as gr
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import zipfile
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


os.chdir("deepfake-detection")

with zipfile.ZipFile("examples.zip","r") as zip_ref:
    zip_ref.extractall(".")

print("Downloading Transfomer models")

transformer_processor = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
transformer_model = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(select_largest=False, post_process=False, device=DEVICE).to(DEVICE).eval()
cnn_model = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=1, device=DEVICE)

checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=DEVICE)
cnn_model.load_state_dict(checkpoint['model_state_dict'])
cnn_model.to(DEVICE).eval()

EXAMPLES_FOLDER = 'examples'
examples_names = os.listdir(EXAMPLES_FOLDER)
examples = [{'path': os.path.join(EXAMPLES_FOLDER, example_name), 'label': example_name.split('_')[0]} for example_name in examples_names]
np.random.shuffle(examples)

def detect_deepfake_transformer(input_image: Image.Image, true_label: str):
    try:
        inputs = transformer_processor(images=input_image, return_tensors="pt")
        with torch.no_grad():
            outputs = transformer_model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()
        confidence = torch.sigmoid(outputs.logits).max().item()
        confidences = {'Real': confidence, 'Fake': 1 - confidence}
        result = "Fake" if predicted_label == 1 else "Real"
        return {"result": result, "confidence": confidences}
    except Exception as e:
        return {"result": "Error", "confidence": str(e)}

def predict_cnn(input_image: Image.Image, true_label: str):
    try:
        face = mtcnn(input_image)
        if face is None:
            return {'Error': 'No face detected'}

        face = face.unsqueeze(0)
        face = torch.nn.functional.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)

        face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy().astype('uint8')
        face = face.to(DEVICE).to(torch.float32) / 255.0

        cam = GradCAM(model=cnn_model, target_layers=[cnn_model.block8.branch1[-1]])
        targets = [ClassifierOutputTarget(0)]
        grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)[0, :]
        visualization = show_cam_on_image(face_image_to_plot / 255.0, grayscale_cam, use_rgb=True)
        face_with_mask = cv2.addWeighted(face_image_to_plot, 1, visualization, 0.5, 0)

        with torch.no_grad():
            output = torch.sigmoid(cnn_model(face)).squeeze(0)
            prediction = "Real" if output.item() < 0.5 else "Fake"
            real_prediction = 1 - output.item()
            fake_prediction = output.item()
            confidences = {'Real': real_prediction, 'Fake': fake_prediction}

        return confidences, prediction, face_with_mask
    except Exception as e:
        return {'Error': str(e)}

def combined_interface(image: Image.Image, true_label: str):
    transformer_result = detect_deepfake_transformer(image, true_label)
    cnn_result, cnn_prediction, cnn_explanation = predict_cnn(image, true_label)

    transformer_confidences = transformer_result['confidence'] if isinstance(transformer_result['confidence'], dict) else {'Real': 0.0, 'Fake': 0.0}
    cnn_confidences = cnn_result if isinstance(cnn_result, dict) else {'Real': 0.0, 'Fake': 0.0}
    cnn_explanation_image = cnn_explanation if isinstance(cnn_explanation, np.ndarray) else None

    combined_confidences = {
        'Real': (transformer_confidences['Real'] + cnn_confidences['Real']) / 2,
        'Fake': (transformer_confidences['Fake'] + cnn_confidences['Fake']) / 2
    }

    final_result = "Real" if combined_confidences['Real'] > combined_confidences['Fake'] else "Fake"
    final_confidence = max(combined_confidences['Real'], combined_confidences['Fake'])

    return combined_confidences, cnn_explanation_image

iface = gr.Interface(
    fn=combined_interface,
    inputs=[
        gr.components.Image(label="Input Image", type="pil"),
        gr.components.Text(label="True Label"),
    ],
    outputs=[
        gr.components.Label(label="Transformer and CNN Confidence score"),
        gr.components.Image(label="CNN Explanation", type="numpy")
    ],
    examples=[[examples[i]["path"], examples[i]["label"]] for i in range(10)],
    cache_examples=False
)

if __name__ == "__main__":
    iface.launch(share=True)