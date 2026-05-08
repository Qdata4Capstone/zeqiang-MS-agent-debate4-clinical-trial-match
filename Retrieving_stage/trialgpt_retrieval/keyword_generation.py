__author__ = "qiao"

"""
generate the search keywords for each patient
"""

import json
import os
import sys
import urllib.error
import urllib.request


DEFAULT_MODEL = "qwen-2.5:7b-instruct"
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_keyword_generation_messages(note):
	system = 'You are a helpful assistant and your task is to help search relevant clinical trials for a given patient description. Please first summarize the main medical problems of the patient. Then generate up to 32 key conditions for searching relevant clinical trials for this patient. The key condition list should be ranked by priority. Please output only a JSON dict formatted as Dict{{"summary": Str(summary), "conditions": List[Str(condition)]}}.'

	prompt =  f"Here is the patient description: \n{note}\n\nJSON output:"

	messages = [
		{"role": "system", "content": system},
		{"role": "user", "content": prompt}
	]
	
	return messages


def generate_with_ollama(model, messages, base_url=DEFAULT_BASE_URL):
	system = ""
	prompt_parts = []
	for message in messages:
		if message["role"] == "system":
			system = message["content"]
		else:
			prompt_parts.append(message["content"])

	payload = {
		"model": model,
		"system": system,
		"prompt": "\n\n".join(prompt_parts),
		"stream": False,
		"format": "json",
		"options": {
			"temperature": 0,
		},
	}

	request = urllib.request.Request(
		f"{base_url.rstrip('/')}/api/generate",
		data=json.dumps(payload).encode("utf-8"),
		headers={"Content-Type": "application/json"},
		method="POST",
	)

	try:
		with urllib.request.urlopen(request, timeout=300) as response:
			body = json.loads(response.read().decode("utf-8"))
	except urllib.error.URLError as exc:
		raise RuntimeError(
			f"Failed to connect to Ollama at {base_url}. "
			"Make sure `ollama serve` is running."
		) from exc

	output = body.get("response", "").strip()
	if not output:
		raise RuntimeError("Ollama returned an empty response.")

	return output


if __name__ == "__main__":
	# the corpus: trec_2021, trec_2022, or sigir
	corpus = sys.argv[1]

	# the model index to use
	model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

	outputs = {}
	os.makedirs("results", exist_ok=True)
	
	with open(f"dataset/{corpus}/queries.jsonl", "r") as f:
		for line in f.readlines():
			entry = json.loads(line)
			messages = get_keyword_generation_messages(entry["text"])

			output = generate_with_ollama(model=model, messages=messages)
			output = output.strip("`").strip("json")
			
			outputs[entry["_id"]] = json.loads(output)

			with open(f"results/retrieval_keywords_{model}_{corpus}.json", "w") as f:
				json.dump(outputs, f, indent=4)
