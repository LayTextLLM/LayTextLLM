import pytesseract
from PIL import Image
import json

def format_ocr_result(data):
    formatted_data = []

    for item in data:
        result = {
            "source": item.get("source", ""),
            "ocr": item.get("ocr", []),
            "poly": item.get("poly", []),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "img_size": item.get("img_size", {}),
            "metadata": json.loads(item.get("metadata", "{}"))
        }
        formatted_data.append(result)

    return formatted_data

# Function to perform OCR and extract bounding boxes
def perform_ocr_with_boxes(image_path):
    # Use Tesseract to extract detailed OCR data
    ocr_data = pytesseract.image_to_data(Image.open(image_path), output_type=pytesseract.Output.DICT)
    
    # Extract text and bounding box information
    results = []
    for i in range(len(ocr_data["text"])):
        if ocr_data["text"][i].strip():  # Ignore empty text entries
            results.append({
                "text": ocr_data["text"][i],
                "bounding_box": [
                    ocr_data["left"][i],
                    ocr_data["top"][i],
                    ocr_data["left"][i] + ocr_data["width"][i],
                    ocr_data["top"][i],
                    ocr_data["left"][i] + ocr_data["width"][i],
                    ocr_data["top"][i] + ocr_data["height"][i],
                    ocr_data["left"][i],
                    ocr_data["top"][i] + ocr_data["height"][i]
                ]
            })
    return results

# Example: Performing OCR with bounding boxes
def process_image_with_ocr_and_boxes(image_path, metadata, question):
    ocr_results = perform_ocr_with_boxes(image_path)
    ocr_texts = [result["text"] for result in ocr_results]
    bounding_boxes = [result["bounding_box"] for result in ocr_results]
    
    data = [
        {
            "source": "custom_ocr",
            "ocr": ocr_texts,
            "poly": bounding_boxes,
            "question": question,
            "answer": "",  # Answer can be added based on processing logic
            "img_size": {"h": Image.open(image_path).height, "w": Image.open(image_path).width},
            "metadata": json.dumps(metadata),
        }
    ]
    return format_ocr_result(data)

# Example image and metadata
image_path = "funsd_82092117.png"
metadata = {"sample_name": "example_sample", "image": "example_image.png", "question_id": "example_qid"}
question = "What is the content in the 'TO:' field?"

# Process the image and print formatted result
formatted_result = process_image_with_ocr_and_boxes(image_path, metadata, question)


json.dump(formatted_result, open("demo_example.json","w", encoding='utf-8'), ensure_ascii=False, indent=4)