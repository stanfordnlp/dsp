from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def openai_to_hf(**kwargs):
    hf_kwargs = {}
    for k, v in kwargs.items():
        if k == "n":
            hf_kwargs["num_return_sequences"] = v
        elif k == "frequency_penalty":
            hf_kwargs["repetition_penalty"] = 1.0 - v
        elif k == "presence_penalty":
            hf_kwargs["diversity_penalty"] = v
        else:
            hf_kwargs[k] = v

    return hf_kwargs


class FlanT5:
    def __init__(self, model="google/flan-t5-large"):
        self.kwargs = {
            "temperature": 0.0,
            "max_new_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
        }

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.history = []

    def basic_request(self, prompt, **kwargs):
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        response = self._generate(prompt, **kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    def request(self, prompt, **kwargs):
        return self.basic_request(prompt, **kwargs)

    def _generate(self, prompt, **kwargs):
        # TODO: Add caching
        kwargs = openai_to_hf(**kwargs)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, **kwargs)
        completions = [
            {"text": c}
            for c in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ]
        response = {
            "prompt": prompt,
            "choices": completions,
        }
        return response

    def print_green(self, text, end="\n"):
        print("\x1b[32m" + str(text) + "\x1b[0m", end=end)

    def print_red(self, text, end="\n"):
        print("\x1b[31m" + str(text) + "\x1b[0m", end=end)

    def inspect_history(self, n=1):
        last_prompt = None
        printed = []

        for x in reversed(self.history[-100:]):
            prompt = x["prompt"]

            if prompt != last_prompt:
                printed.append((prompt, x["response"]["choices"]))

            last_prompt = prompt

            if len(printed) >= n:
                break

        for prompt, choices in reversed(printed):
            print("\n\n\n")
            print(prompt, end="")
            self.print_green(choices[0]["text"], end="")

            if len(choices) > 1:
                self.print_red(f" \t (and {len(choices)-1} other completions)", end="")
            print("\n\n\n")

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        # TODO: Handle only_completed=True

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        if kwargs.get("n", 1) > 1:
            kwargs["num_beams"] = max(5, kwargs["n"])

        response = self.request(prompt, **kwargs)
        return [c["text"] for c in response["choices"]]


if __name__ == "__main__":
    model = FlanT5()
    response = model("Who was the first man to walk on the moon?\nFinal answer: ")
    print(response)
