# process_transcript.py
# Convert a transcript file to a JSONL file for fine-tuning.

import json
import argparse

def process_transcript(file_path):
    """
    Reads a transcript file and converts each dialogue turn into a dictionary.
    Assumes each line is of the form 'Speaker: Dialogue'.
    If a line does not include a colon, it assigns a default speaker label.
    """
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip blank lines
            # Look for a colon to split speaker and dialogue
            if ':' in line:
                speaker, dialogue = line.split(':', 1)
                record = {
                    "speaker": speaker.strip(),
                    "utterance": dialogue.strip()
                }
            else:
                # If no colon, treat it as narration or continuation
                record = {
                    "speaker": "NARRATOR",
                    "utterance": line
                }
            records.append(record)
    return records

def write_jsonl(records, output_path):
    """
    Writes the list of records to a JSONL file where each line is a JSON object.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            json_line = json.dumps(record, ensure_ascii=False)
            f.write(json_line + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert TV show transcript to JSONL for fine-tuning using ollama.")
    parser.add_argument("input_file", help="Path to the input transcript file")
    parser.add_argument("output_file", help="Path to the output JSONL file")
    args = parser.parse_args()

    # Process the transcript and write JSONL output
    transcript_records = process_transcript(args.input_file)
    write_jsonl(transcript_records, args.output_file)
    
    print(f"Processed transcript saved to {args.output_file}")

