"""
Use an LLM to perform binary classification on samples, outputting either `<judgment>YES</judgment>` or `<judgment>NO</judgment>`. The classification criteria are determined by the provided prompt.

The input can be any data type supported by `data_loader`. 

The output directory contains multiple chunks, each of which contains one or more Arrow files storing the binary classification results. The classification results for each sample are represented as an array, whose length equals the number of evaluation attempts. Each entry in the array corresponds to one model response.

If the output directory already contains chunks, inference will resume from the last completed checkpoint.
"""

import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )
)

import argparse
from gandolf.llms.LLM_Router import create_LLM_instance
from gandolf.util.utils import create_dir, write_list_to_disk_as_hf_dataset
from timeit import default_timer as timer
import logging
from data_loader import DataLoader
from pathlib import Path
from gandolf.inference.response_checker.binary_classification_checker import (
    response_checker_thinking_lenient,
    response_checker_thinking_strict,
    response_checker_no_thinking_lenient,
    response_checker_no_thinking_strict
)


class BinaryClassifier:

    def __init__(
            self,
            input_path,
            output_dir,
            generations,
            model_name,
            batch_size,
            max_output_tokens,
            url,
            concurrency,
            thinking,
            rigor_validation,
            apply_chat_template
    ):
        self.input_path = input_path
        self.output_dir = output_dir

        output_folder = Path(self.output_dir)
        output_folder.mkdir(parents=True, exist_ok=True)

        self.chunk_counter = len(os.listdir(self.output_dir))
        print(f"chunk_counter = {self.chunk_counter}")
        self.batch_size = batch_size

        self.generations = generations

        if thinking:
            if rigor_validation:
                response_checker = response_checker_thinking_strict
            else:
                response_checker = response_checker_thinking_lenient
        else:
            if rigor_validation:
                response_checker = response_checker_no_thinking_strict
            else:
                response_checker = response_checker_no_thinking_lenient

        self.model = create_LLM_instance(
            model_name,
            url,
            concurrency,
            max_output_tokens,
            response_checker,
            self.generations,
            apply_chat_template
        )

        self.existing_samples = self.chunk_counter * self.batch_size
        self.samples_counter = 0
        self.prompts = []
        self.distinct_prompts = []
        self.prompts_counter = 0
        self.bad_response_count = 0

    def compose_prompt(self, sample, idx):
        self.samples_counter += 1

        if self.samples_counter > self.existing_samples:

            prompt = sample["prompt"]

            self.distinct_prompts.append(prompt)

            for i in range(self.generations):
                self.prompts.append(prompt)
                self.prompts_counter += 1

            if self.samples_counter % self.batch_size == 0:
                # Write a batch of samples to disk and release memory
                result = self.batch_inference()
                self.prompts.clear()
                self.distinct_prompts.clear()
                self.chunk_counter += 1
                chunk_dir_path = os.path.join(self.output_dir, "chunk" + str(self.chunk_counter))
                create_dir(chunk_dir_path)
                write_list_to_disk_as_hf_dataset(chunk_dir_path, result)
                result.clear()
                del result

        return sample

    def batch_inference(self):
        retry = 0
        prompts = self.prompts
        indices = list(range(len(prompts)))
        responses = [None] * len(prompts)

        while retry < 10 and len(prompts) > 0:
            print(f"Start inferring for {len(prompts)} prompts")  # time in seconds
            start = timer()
            self.model.set_n(1 if retry > 0 else self.generations)
            outputs = self.model.inference(prompts if retry > 0 else self.distinct_prompts)
            end = timer()
            print(f"Binary classification inference takes {end - start}s")  # time in seconds

            if retry == 0 and self.generations > 1:
                # flatten the 2D output (e.g., [[sample1_g1, sample1_g2, sample1_g3], [sample2_g1, sample2_g2, sample2_g3], ...]) to 1D
                outputs = [x for row in outputs for x in row]
                if len(outputs) != len(self.distinct_prompts) * self.generations:
                    print(f"outputs length ({len(outputs)}) should be the same as inputs length ({len(prompts)})")
                    exit(0)

            invalid_prompts = []
            invalid_indices = []
            # Verify whether the responses are valid
            for i in range(0, len(prompts)):
                if outputs[i] is not None:
                    if responses[indices[i]] is not None:
                        print(f"{indices[i]}th response should be None.")
                        exit(0)
                    responses[indices[i]] = outputs[i]
                else:
                    invalid_prompts.append(prompts[i])
                    invalid_indices.append(indices[i])

            prompts = invalid_prompts
            indices = invalid_indices
            retry += 1

        if len(prompts) > 0:
            self.bad_response_count += len(prompts)
            print(f"There are {len(prompts)} invalid responses even after 10 tries")

        classifications = [None] * self.generations
        results = []

        for i in range(len(self.prompts)):
            if i != 0 and i % self.generations == 0:
                results.append({"runs": classifications})
                classifications = [None] * self.generations

            classifications[i % self.generations] = responses[i]

        results.append({"runs": classifications})

        return results

    def process(self):
        dl = DataLoader(self.input_path)
        ds = dl.load_dataset()

        updated_dataset = ds.map(self.compose_prompt, with_indices=True)

        loop = 0
        for example in updated_dataset:
            # loop over the dataset to activate the mapped function compose_prompt. We need to do so because streaming mode is open
            loop += 1

        if len(self.prompts) > 0:
            result = self.batch_inference()
            self.prompts.clear()
            self.distinct_prompts.clear()
            self.chunk_counter += 1
            chunk_dir_path = os.path.join(self.output_dir, "chunk" + str(self.chunk_counter))
            create_dir(chunk_dir_path)
            write_list_to_disk_as_hf_dataset(chunk_dir_path, result)
            result.clear()
            del result

        difference = 0 if self.samples_counter - self.existing_samples < 0 else self.samples_counter - self.existing_samples

        if self.prompts_counter != difference * self.generations:
            print("Unexpected prompt set size = " + str(self.prompts_counter))
            exit(1)

        if loop != self.samples_counter:
            print("Unexpected loop size = " + str(loop))
            exit(1)

        logging.warning("#bad responses = " + str(self.bad_response_count))
        logging.warning("#prompts = " + str(self.prompts_counter))
        logging.warning("#bad responses ratio = " + "0" if self.prompts_counter == 0 else str(self.bad_response_count * 1.0 / self.prompts_counter))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, default="")
    parser.add_argument("-o", "--output_dir", type=str, default="")
    parser.add_argument("-g", "--generations", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1000)  # chunk size
    parser.add_argument("--max_output_tokens", type=int, default=8192)
    parser.add_argument("--url", type=str, default=None)  # vllm serve url
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--is_thinking", action="store_true")
    parser.add_argument("--need_rigor_validation", action="store_true")
    parser.add_argument("--apply_chat_template", action="store_true")  # Don't set if the prompt is already formatted by a predefined template
    args = parser.parse_args()
    print(args)

    classifier = BinaryClassifier(
        input_path=args.input_path,
        output_dir=args.output_dir,
        generations=args.generations,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_output_tokens=args.max_output_tokens,
        url=args.url,
        concurrency=args.concurrency,
        thinking=args.is_thinking,
        rigor_validation=args.need_rigor_validation,
        apply_chat_template=args.apply_chat_template
    )
    classifier.process()

