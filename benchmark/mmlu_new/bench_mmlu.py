import argparse
import json
import os
import time

import numpy as np
import openai
import pandas as pd

from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import print_highlight, terminate_process, wait_for_server

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    return " ".join(subject.split("_"))


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def evaluate_mmlu(client, model, arguments, labels, parallel):
    preds = []
    total = len(arguments)

    for i, arg in enumerate(arguments):
        prompt = arg["examples"] + arg["question"]
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1,
            )
            answer = response.choices[0].message.content.strip()
            pred = answer[0] if len(answer) > 0 else ""
        except Exception as e:
            print(f"Error on question {i}: {e}")
            pred = ""
        preds.append(pred)

        if (i + 1) % 100 == 0:
            print(f"Progress: {i + 1}/{total}")

    return preds


def main(args):
    # Launch server
    server_process, port = launch_server_cmd(
        f"python3 -m sglang.launch_server --model-path {args.model_path} "
        f"--host 0.0.0.0 --log-level warning"
    )

    try:
        wait_for_server(f"http://localhost:{port}")
        print_highlight(f"Server ready on port {port}")

        # Create OpenAI client
        client = openai.Client(
            base_url=f"http://127.0.0.1:{port}/v1", api_key="None"
        )

        # Discover subjects
        subjects = sorted(
            [
                f.split("_test.csv")[0]
                for f in os.listdir(os.path.join(args.data_dir, "test"))
                if "_test.csv" in f
            ]
        )

        # Build prompts
        arguments = []
        labels = []
        num_questions = []

        for subject in subjects[: args.nsub]:
            dev_df = pd.read_csv(
                os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
            )[: args.ntrain]
            test_df = pd.read_csv(
                os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
            )
            num_questions.append(test_df.shape[0])

            k = args.ntrain
            few_shot_examples = gen_prompt(dev_df, subject, k)

            for i in range(test_df.shape[0]):
                prompt_end = format_example(test_df, i, include_answer=False)
                arguments.append(
                    {
                        "examples": few_shot_examples,
                        "question": prompt_end,
                    }
                )
                label = test_df.iloc[i, test_df.shape[1] - 1]
                labels.append(label)

        print_highlight(
            f"Loaded {len(arguments)} questions across {min(args.nsub, len(subjects))} subjects"
        )

        # Run evaluation
        tic = time.perf_counter()
        preds = evaluate_mmlu(client, args.model_path, arguments, labels, args.parallel)
        latency = time.perf_counter() - tic

        # Compute accuracy
        cors = [pred == label for pred, label in zip(preds, labels)]

        pt = 0
        for subject, num_qs in zip(subjects[: args.nsub], num_questions):
            print(
                f"subject: {subject}, #q:{num_qs}, acc: {np.mean(cors[pt : pt + num_qs]):.3f}"
            )
            pt += num_qs
        assert pt == len(cors)
        weighted_acc = np.mean(cors)

        # Print results
        print_highlight(f"Total latency: {latency:.3f}s")
        print_highlight(f"Average accuracy: {weighted_acc:.3f}")

        # Write results
        with open(args.result_file, "a") as fout:
            value = {
                "task": "mmlu",
                "backend": "sglang_openai",
                "model": args.model_path,
                "num_gpus": 1,
                "latency": round(latency, 3),
                "accuracy": round(weighted_acc, 3),
                "num_requests": len(arguments),
                "other": {
                    "nsub": args.nsub,
                    "parallel": args.parallel,
                },
            }
            fout.write(json.dumps(value) + "\n")

    finally:
        terminate_process(server_process)
        print_highlight("Server terminated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, default="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    )
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data-dir", "-d", type=str, default="data")
    parser.add_argument("--nsub", type=int, default=60)
    parser.add_argument("--parallel", type=int, default=64)
    parser.add_argument("--result-file", type=str, default="result.jsonl")
    args = parser.parse_args()
    main(args)
