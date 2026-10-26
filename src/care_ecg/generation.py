from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .types import RetrievedFact


class TextGenerator(Protocol):
    def generate(self, prompt: str, facts: Sequence[RetrievedFact]) -> str: ...


class TemplateGenerator:
    """Deterministic offline generator whose claims are restricted to supplied evidence."""

    def generate(self, prompt: str, facts: Sequence[RetrievedFact]) -> str:
        if not facts:
            return (
                "Insufficient retrieved evidence is available to provide a grounded ECG "
                "interpretation. Clinical review is required."
            )
        fields = {}
        for line in prompt.splitlines():
            for key in (
                "Patient Query",
                "Predicted Diagnosis",
                "Diagnosis Posterior",
                "Key Causal Factors (from VAE/Graph)",
                "Attribution Scope",
                "Counterfactual Check",
            ):
                prefix = key + ":"
                if line.startswith(prefix):
                    fields[key] = line[len(prefix) :].strip()
        lines = ["Evidence-grounded assessment:"]
        if fields.get("Patient Query"):
            lines.append(f"- Question addressed: {fields['Patient Query']}")
        diagnosis = fields.get("Predicted Diagnosis")
        posterior = fields.get("Diagnosis Posterior")
        if diagnosis and posterior:
            probabilities = {}
            for item in posterior.split(","):
                label, separator, value = item.strip().partition("=")
                if separator:
                    probabilities[label.strip()] = value.strip()
            if diagnosis in probabilities:
                lines.append(
                    f"- Model confidence for the stated {diagnosis} class: "
                    f"{probabilities[diagnosis]}."
                )
        causal = fields.get("Key Causal Factors (from VAE/Graph)")
        if causal and causal != "None":
            scope = fields.get("Attribution Scope", "reported fused posterior")
            qualifier = (
                "Bayesian-component latent sensitivity factors"
                if scope == "bayesian component"
                else "Reported-posterior latent sensitivity factors"
            )
            lines.append(f"- {qualifier}: {causal}.")
        counterfactual = fields.get("Counterfactual Check")
        if counterfactual and counterfactual != "Not requested.":
            lines.append(f"- Counterfactual sensitivity: {counterfactual}")
        for index, fact in enumerate(facts, start=1):
            lines.append(f"- {fact.text} [Fact {index}]")
        lines.append(
            "The probabilistic output and observational latent-factor sensitivity are model "
            "findings, not physiological interventions or a substitute for direct waveform "
            "measurements and clinician review."
        )
        return "\n".join(lines)


class OpenAIResponsesGenerator:
    """Optional OpenAI Responses API adapter.

    ``store=False`` is explicit because prompts may contain clinical context. Whether
    this is sufficient for a deployment's privacy requirements is the operator's decision.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.3,
        max_tokens: int = 600,
        client=None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI generation requires `pip install 'care-ecg[openai]'`"
            ) from exc
        self.client = client or OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, facts: Sequence[RetrievedFact]) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=(
                "You are the CARE-ECG Response Agent. Use only the supplied retrieved facts "
                "for clinical factual claims. Cite each claim with its [Fact n] tag. State "
                "uncertainty and never invent waveform measurements."
            ),
            input=prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            store=False,
        )
        return response.output_text


class HuggingFaceGenerator:
    """Optional local causal-language-model adapter, including LLaMA-style chat models."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 600,
        device_map: str = "auto",
    ):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Local generation requires `pip install 'care-ecg[local-llm]'`"
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, device_map=device_map)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, facts: Sequence[RetrievedFact]) -> str:
        import torch

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the CARE-ECG Response Agent. Ground every clinical factual "
                    "claim in a supplied [Fact n], cite it, and do not invent measurements."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            rendered = messages[0]["content"] + "\n\n" + prompt
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        do_sample = self.temperature > 0
        generation_options = {
            "max_new_tokens": self.max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_options["temperature"] = self.temperature
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_options)
        continuation = generated[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(continuation, skip_special_tokens=True).strip()
