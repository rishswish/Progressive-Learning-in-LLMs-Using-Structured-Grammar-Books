# Multichoice

1. Raw Processing

run ```python lesson_ocr.py <input_image_dir> <output_result_dir>```

for example ```python lesson_ocr.py textbook textbook```

2. Manually Checking

><li> check the "multi-choice" in output json files, to make sure replace the places that needs to insert answer with "______"(exact 6 underscores)
><li>  check if there are omitting of the questions, it is 12 questions for each lesson
><li>  check and correct other things that you think is reasonable

3. Create Docx Results

run ```python post_lesson_ocr.py <output_result_dir> <output_result_dir>```

for example ```python post_lesson_ocr.py textbook textbook```

>>If the number of multi-choice question is not 12, the code will throw an error, be aware of it and modifiy the input json if it throws error. And correct other errors if needed.