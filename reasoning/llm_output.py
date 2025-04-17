import csv
import json
import os
import concurrent.futures
import threading
from queue import Queue, Empty
from typing import List, Dict, Any, Tuple
import openai
from tqdm import tqdm


def load_json_data(file_path: str):
    """Simple JSON data loading"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error, position: line {e.lineno}, column {e.colno}")
        print(f"Error message: {str(e)}")
        return None


def create_openai_client(base_url: str, api_key: str = "sk-"):
    """Create OpenAI client"""
    return openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def call_llm_api(item: Dict[str, Any], client_info: Tuple[openai.OpenAI, str]) -> tuple:
    """Call the model using OpenAI API"""
    client, model_name = client_info
    filepath = item['filepath']
    prompt = item['conversations'][0]['value']

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            timeout=300.0,
            # max_tokens=12000
        )
        return filepath, response.choices[0].message.content
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return filepath, ""


def json_2_csv_double_vllm(ports, json_file: str, csv_file: str, base_urls, base_models, batch_size: int, max_thread: int):
    # Configure parameters for multiple machines
    machines = [{"base_url": f"http://{url}:{port}/v1/", "model_name": base_model} for url, base_model, port in zip(base_urls, base_models, ports)]
    # Create multiple clients
    clients = [(create_openai_client(machine["base_url"]), machine["model_name"]) for machine in machines]

    data = load_json_data(json_file)
    if not data:
        return

    csv_headers = ['filepath', 'sourceCode', 'translateCode']

    # Get data that needs to be reprocessed
    incomplete_filepaths = set()
    # Check if CSV file exists
    if os.path.exists(csv_file):
        try:
            with open(csv_file, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row['translateCode'] or len(row['translateCode']) < 1:
                        incomplete_filepaths.add(row['filepath'])
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return

        # Filter out data that needs to be processed
        incomplete_data = [item for item in data if item['filepath'] in incomplete_filepaths]
        print(f"Found {len(incomplete_data)} items that need to be reprocessed")
    else:
        # If CSV file doesn't exist, all data needs to be processed
        incomplete_data = data
        print(f"CSV file doesn't exist, will process all {len(incomplete_data)} items")

        # Create an empty CSV file with headers only
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            # Fill in initial data with only filepath and sourceCode
            initial_rows = []
            for item in data:
                # Get sourceCode, assuming it's in the first element of the conversations array
                source_code = ""
                if "sourceCode" in item:
                    source_code = item["sourceCode"]
                elif "conversations" in item and len(item["conversations"]) > 0:
                    # This may need to be adjusted based on actual data structure
                    if "value" in item["conversations"][1]:
                        source_code = item["conversations"][1]["value"]

                initial_rows.append({
                    'filepath': item['filepath'],
                    'sourceCode': source_code,
                    'translateCode': ''
                })
            writer.writerows(initial_rows)

    if not incomplete_data:
        print("No data needs to be processed")
        return

    # Create a result queue for safe CSV writing
    result_queue = Queue()
    # Create CSV writer thread
    csv_writer_thread = threading.Thread(target=csv_writer_worker, args=(csv_file, result_queue, csv_headers))
    csv_writer_thread.daemon = True
    csv_writer_thread.start()

    # Sort by input_output_Token_Length
    incomplete_data.sort(key=lambda x: x.get('input_output_Token_Length', 0))
    print(f"Data has been sorted by input_output_Token_Length")

    # Pre-divide all batches
    batches = []
    for i in range(0, len(incomplete_data), batch_size):
        batches.append(incomplete_data[i:i + batch_size])

    print(f"Divided into a total of {len(batches)} batches")

    # Create batch queue and result counter
    batch_queue = Queue()
    for i, batch in enumerate(batches):
        batch_queue.put((i, batch))

    # Create threads to process batches
    url_threads = []
    for i, client in enumerate(clients):
        thread = threading.Thread(target=process_batches_for_url,
                                  args=(client, batch_queue, result_queue, max_thread, machines[i]))
        thread.daemon = True
        url_threads.append(thread)
        thread.start()

    # Wait for all URL threads to complete
    for thread in url_threads:
        thread.join()

    # Send termination signal to CSV write thread
    result_queue.put(None)
    # Wait for CSV write thread to complete
    csv_writer_thread.join()
    print("All processing completed")


def process_batches_for_url(client, batch_queue, result_queue, max_thread, url_id):
    """Thread function for each URL to process batches"""
    while not batch_queue.empty():
        try:
            # Try to get a batch
            batch_idx, batch = batch_queue.get_nowait()
        except Empty:
            break

        print(f"{url_id['base_url']} processing batch {batch_idx + 1}, containing {len(batch)} items...")

        # Use multi-threading within the same batch of data
        max_workers = min(max_thread, len(batch))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {}
            for item in batch:
                future_to_item[executor.submit(call_llm_api, item, client)] = item

            # Use tqdm to track thread completion
            for future in tqdm(concurrent.futures.as_completed(future_to_item),
                               total=len(future_to_item),
                               desc=f"{url_id['base_url']} batch {batch_idx + 1}"):
                try:
                    filepath, result = future.result()
                    if result:
                        # Put result in queue
                        result_queue.put((filepath, result))
                    else:
                        print(f"Processing file {filepath} returned no result")
                except Exception as e:
                    item = future_to_item[future]
                    print(f"Error processing file {item['filepath']}: {e}")

        batch_queue.task_done()
        print(f"{url_id['base_url']} completed batch {batch_idx + 1} processing")


def csv_writer_worker(csv_file: str, result_queue: Queue, csv_headers: list):
    """CSV writer worker thread"""
    buffer = []
    buffer_size = 50

    while True:
        # Get result from queue
        result = result_queue.get()
        if result is None:  # Termination signal
            # Save remaining results in buffer
            if buffer:
                save_results_to_csv(csv_file, buffer, csv_headers)
            break

        # Add to buffer
        buffer.append(result)

        # Save results when buffer size is reached or queue is empty
        if len(buffer) >= buffer_size:
            print(f"Saving {len(buffer)} results to CSV file")
            save_results_to_csv(csv_file, buffer, csv_headers)
            buffer = []

        result_queue.task_done()


def save_results_to_csv(csv_file: str, results: list, csv_headers: list):
    """Save results to CSV file"""
    try:
        # Read current CSV content
        csv_data = []
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            csv_data = list(reader)

        # Update results
        for filepath, translate_code in results:
            for row in csv_data:
                if row['filepath'] == filepath:
                    row['translateCode'] = translate_code
                    break

        # Save updated CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            writer.writerows(csv_data)

    except Exception as e:
        print(f"Error updating CSV file: {e}")