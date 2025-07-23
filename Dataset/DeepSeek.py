import json
import time
from openai import OpenAI
from typing import List, Dict, Any
import random
import requests

class QADatasetGenerator:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        
    def generate_multiple_choice_prompt(self, url: str, content: str) -> str:
        """Generate prompt for multiple choice questions"""
        prompt = f"""Based on the following web page content, generate exactly 3 multiple-choice questions with 4 options each. The questions should test understanding of the content and be factually accurate.

URL: {url}
Content: {content}

Requirements:
1. Generate exactly 3 multiple-choice questions
2. Each question must have exactly 4 options labeled (A), (B), (C), (D)
3. Questions should be diverse and cover different aspects of the content
4. All options should be plausible but only one should be correct
5. Questions should be clear and specific
6. Focus on factual information present in the content

Format your response as a JSON array with this exact structure:
[
    {{
        "question": "What is the main focus of the EECS department?",
        "options": {{
            "A": "Only electrical engineering",
            "B": "Computer science and electrical engineering", 
            "C": "Only mathematics",
            "D": "Only physics"
        }},
        "correct_answer": "B"
    }}
]

Generate 3 questions now:"""
        return prompt

    def generate_open_ended_prompt(self, url: str, content: str) -> str:
        """Generate prompt for open-ended questions"""
        prompt = f"""Based on the following web page content, generate exactly 3 open-ended questions with comprehensive answers. The questions should test deeper understanding of the content.

URL: {url}
Content: {content}

Requirements:
1. Generate exactly 3 open-ended questions
2. Each answer should be comprehensive (2-4 sentences)
3. Questions should be diverse and cover different aspects of the content
4. Focus on factual information present in the content
5. Questions can ask about explanations, descriptions, comparisons, or analysis
6. Answers should be based solely on the provided content

Format your response as a JSON array with this exact structure:
[
    {{
        "question": "What makes the EECS department at UC Berkeley distinctive?",
        "answer": "The EECS department at UC Berkeley is distinctive because it offers one of the strongest research and instructional programs in the field worldwide, with top-ranked programs that attract stellar students and professors from around the world who pioneer the frontiers of information science and technology."
    }}
]

Generate 3 questions now:"""
        return prompt

    def call_api(self, prompt: str, max_retries: int = 3) -> str:
        """Call NVIDIA NIM API using requests and LLaMA-4 Maverick model"""
        invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Authorization": "Bearer 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",
            "Accept": "application/json"
        }

        payload = {
            "model": "meta/llama-4-maverick-17b-128e-instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that generates structured QA dataset."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream": False
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(invoke_url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def parse_json_response(self, response: str) -> List[Dict]:
        """Parse JSON response with error handling"""
        try:
            # Try to find JSON in the response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                print(f"No JSON array found in response: {response}")
                return []
        
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response was: {response}")
            return []

    def generate_multiple_choice_qa(self, url: str, content: str) -> List[Dict]:
        """Generate multiple choice Q&A pairs"""
        prompt = self.generate_multiple_choice_prompt(url, content)
        
        try:
            response = self.call_api(prompt)
            questions = self.parse_json_response(response)
            
            # Format for output
            formatted_qa = []
            for q in questions:
                if all(key in q for key in ['question', 'options', 'correct_answer']):
                    # Format options as string
                    options_str = f"(A) {q['options']['A']} (B) {q['options']['B']} (C) {q['options']['C']} (D) {q['options']['D']}"
                    
                    formatted_qa.append({
                        "chunk_url": url,
                        "content": content,
                        "question": f"{q['question']} {options_str}",
                        "answer": f"({q['correct_answer']})"
                    })
            
            return formatted_qa
        
        except Exception as e:
            print(f"Error generating multiple choice Q&A for {url}: {e}")
            return []

    def generate_open_ended_qa(self, url: str, content: str) -> List[Dict]:
        """Generate open-ended Q&A pairs"""
        prompt = self.generate_open_ended_prompt(url, content)
        
        try:
            response = self.call_api(prompt)
            questions = self.parse_json_response(response)
            
            # Format for output
            formatted_qa = []
            for q in questions:
                if all(key in q for key in ['question', 'answer']):
                    formatted_qa.append({
                        "chunk_url": url,
                        "content": content,
                        "question": q['question'],
                        "answer": q['answer']
                    })
            
            return formatted_qa
        
        except Exception as e:
            print(f"Error generating open-ended Q&A for {url}: {e}")
            return []

    def process_dataset(self, input_file: str, output_mc_file: str, output_open_file: str):
        """Process the entire dataset and generate Q&A pairs, saving immediately to file"""
        
        print("Starting Q&A dataset generation...")
        
        # Open output files for writing (append mode)
        with open(output_mc_file, 'w', encoding='utf-8') as f_mc, \
            open(output_open_file, 'w', encoding='utf-8') as f_open:
            
            # Read input dataset
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            total_chunks = len(lines)
            print(f"Processing {total_chunks} chunks...")
            
            for i, line in enumerate(lines, 1):
                try:
                    chunk = json.loads(line.strip())
                    url = chunk['url']
                    content = chunk['text']
                    
                    print(f"\nProcessing chunk {i}/{total_chunks}: {url}")
                    
                    # Generate multiple choice questions
                    print("Generating multiple choice questions...")
                    mc_qa = self.generate_multiple_choice_qa(url, content)
                    for item in mc_qa:
                        self.save_one_line(item, f_mc)
                    print(f"✅ Wrote {len(mc_qa)} multiple choice QA")
                    
                    time.sleep(1)
                    
                    # Generate open-ended questions
                    print("Generating open-ended questions...")
                    open_qa = self.generate_open_ended_qa(url, content)
                    for item in open_qa:
                        self.save_one_line(item, f_open)
                    print(f"✅ Wrote {len(open_qa)} open-ended QA")
                    
                    time.sleep(1)
                
                except Exception as e:
                    print(f"⚠️ Error processing chunk {i}: {e}")
                    continue
            
            print("\n✅ Dataset generation complete.")

    def save_one_line(self, data: Dict, file_handle):
        """Write one JSON object per line to open file handle"""
        file_handle.write(json.dumps(data, ensure_ascii=False) + '\n')

def main():
    # Configuration
    API_KEY = "YOUR_API_KEY"  # Replace with your actual API key
    INPUT_FILE = "dataset.jsonl"
    OUTPUT_MC_FILE = "multiple_choice_qa_dataset.jsonl"
    OUTPUT_OPEN_FILE = "open_ended_qa_dataset.jsonl"
    
    # Create generator instance
    generator = QADatasetGenerator(API_KEY)
    
    # Process the dataset
    generator.process_dataset(INPUT_FILE, OUTPUT_MC_FILE, OUTPUT_OPEN_FILE)

if __name__ == "__main__":
    main()